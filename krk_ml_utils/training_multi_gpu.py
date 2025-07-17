# training_v4.py

import jax
import optax
import os
import cloudpickle
import pandas as pd
import jax.numpy as jnp
from datetime import datetime
from typing import Callable, Tuple, Optional, Dict, Any
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
from krk_ml_utils.datasets import JaxDataLoader # Assuming this is your dataloader

# --- No changes needed to these helper functions ---

def _unpack_leaf(leaf):
    """
    Unpacks nnx.State objects to extract their underlying JAX array values.
    Returns None for leaves that don't contain arrays.
    """
    if isinstance(leaf, nnx.State):
        return leaf.value
    return leaf

def _save_checkpoint_bundle(
    dir_path: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    grad_accumulator: Any,
    epoch: int,
    global_step: int,
):
    """
    Saves the complete training state as a bundle of files in a directory.
    """
    os.makedirs(dir_path, exist_ok=True)

    # GATHER: Pull the sharded state from all devices to the host CPU.
    # This converts distributed arrays back into standard JAX arrays.
    model_state_host = jax.device_get(nnx.state(model))
    optimizer_state_host = jax.device_get(nnx.state(optimizer))
    grad_accumulator_host = jax.device_get(grad_accumulator)

    # Split the model and optimizer to get GraphDef and State separately
    model_graphdef, _ = nnx.split(model)
    optimizer_graphdef, _ = nnx.split(optimizer)

    # Create temporary host-side copies for saving using proper merge
    model_host = nnx.merge(model_graphdef, model_state_host)
    optimizer_host = nnx.merge(optimizer_graphdef, optimizer_state_host)
    optimizer_host.model = model_host  # Re-link optimizer to the host-side model

    # 1. Save model using the host-side copy
    model_path = os.path.join(dir_path, "model.pkl")
    with open(model_path, "wb") as f:
        cloudpickle.dump(model_host, f)

    # 2. Save optimizer state using the host-side copy
    optimizer_path = os.path.join(dir_path, "optimizer.pkl")
    with open(optimizer_path, "wb") as f:
        cloudpickle.dump(optimizer_host, f)

    # 3. Save the rest of the training state (already on host)
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "grad_accumulator": grad_accumulator_host,
    }
    training_state_path = os.path.join(dir_path, "training_state.pkl")
    with open(training_state_path, "wb") as f:
        cloudpickle.dump(training_state, f)

    print(f"Checkpoint bundle saved to {dir_path}")


def _find_latest_checkpoint_dir(checkpoint_dir: str) -> Optional[str]:
    """
    Finds the latest checkpoint bundle directory based on epoch and step numbers.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # List subdirectories that match the expected naming convention
    subdirs = [
        d
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("epoch_")
    ]

    if not subdirs:
        return None

    # Alphanumeric sort on "epoch_xxxx_step_yyyyyyyy" works correctly to find the latest
    latest_subdir = sorted(subdirs)[-1]
    return os.path.join(checkpoint_dir, latest_subdir)

# --- Main Training Function ---


def train_flax_lm(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    loss_fn: Callable,
    train_dataloader: Any, # Using Any for placeholder
    test_dataloader: Optional[Any] = None,
    eval_fn: Optional[Callable] = None,
    num_epochs: int = 10,
    checkpoint_dir: str = "checkpoints",
    save_every_epochs: int = 1,
    save_every_steps: Optional[int] = None,
    log_train_metrics_every_steps: int = 100,
    eval_every_steps: Optional[int] = 1000,
    accumulation_steps: int = 1,
    resume_from_checkpoint: bool = True,
    new_schedule_step: int = None,
) -> Tuple[nnx.Module, Dict[str, list]]:
    """
    Trains a Flax NNX model with gradient accumulation and intra-epoch checkpointing.
    """
    # --- 1. Setup for Data Parallelism ---
    # Create a 1D mesh over all available devices (real or virtual).
    # We name the single axis 'data' for clarity.
    mesh = Mesh(jax.devices(), axis_names=('data',))
    print(f"Using device mesh with {len(jax.devices())} devices: {jax.devices()}")

    # Define the sharding rule for data batch
    # Get a sample batch to determine the sharding structure
    sample_batch = next(iter(train_dataloader))
    
    # Create NamedSharding objects instead of PartitionSpec
    data_sharding = jax.tree_util.tree_map(
        lambda _: NamedSharding(mesh, P('data')), sample_batch
    )

    # For parameters, we'll use a simpler approach - just replicate everything
    def create_replicated_sharding(pytree):
        """Create replicated sharding for any pytree structure containing arrays"""
        def _shard_leaf(x):
            if hasattr(x, 'shape') and hasattr(x, 'dtype'):  # It's an array
                return NamedSharding(mesh, P())  # Replicate
            else:
                return x  # Keep non-arrays as-is
        return jax.tree_util.tree_map(_shard_leaf, pytree)

    # --- JIT-Compiled Helper Functions with Explicit Sharding ---
    @nnx.jit(
        in_shardings=(None, None, data_sharding),  # model=replicated, metrics=replicated, batch=sharded
        out_shardings=(NamedSharding(mesh, P()), None)  # loss=replicated, grads=replicated
    )
    def train_step(
        model: nnx.Module, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Any]:
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)

        if isinstance(batch, dict):
            labels = batch.get("label", batch.get("labels"))
        else:
            _, labels = batch
        metrics.update(loss=loss, logits=logits, labels=labels)
        return loss, grads

    if eval_fn is None:
        eval_fn = loss_fn
        
    @nnx.jit(
        in_shardings=(None, None, data_sharding),  # model=replicated, metrics=replicated, batch=sharded
    )
    def eval_step(
        model: nnx.Module, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray]
    ):
        loss, logits = eval_fn(model, batch)
        if isinstance(batch, dict):
            labels = batch.get("label", batch.get("labels"))
        else:
            _, labels = batch
        metrics.update(loss=loss, logits=logits, labels=labels)
    
    # --- Initialization ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch, global_step, metrics_history = 0, 0, {}

    # Initialize gradient accumulator to None - will be set on first batch
    grad_accumulator = None
    print("Gradient accumulator will be initialized on first batch.")
    
    # --- Resume Logic ---
    if resume_from_checkpoint:
        latest_checkpoint_dir = _find_latest_checkpoint_dir(checkpoint_dir)
        if latest_checkpoint_dir:
            print(f"Found latest checkpoint: {latest_checkpoint_dir}")
            
            # Load the checkpoint (these are already on host)
            model_path = os.path.join(latest_checkpoint_dir, "model.pkl")
            with open(model_path, "rb") as f: 
                loaded_model = cloudpickle.load(f)

            optimizer_path = os.path.join(latest_checkpoint_dir, "optimizer.pkl")
            with open(optimizer_path, "rb") as f: 
                loaded_optimizer = cloudpickle.load(f)

            state_path = os.path.join(latest_checkpoint_dir, "training_state.pkl")
            with open(state_path, "rb") as f: 
                training_state = cloudpickle.load(f)
            
            start_epoch = training_state['epoch']
            global_step = training_state['global_step'] 
            grad_accumulator = training_state['grad_accumulator']

            # Replace the current model and optimizer with loaded ones
            # This ensures we get the exact same architecture and state
            model = loaded_model
            #optimizer = loaded_optimizer
            ###
            if new_schedule_step is not None:
                print("Found new_lr_schedule. Attempting to replace optimizer's LR.")
###
                if (hasattr(loaded_optimizer, 'step') and
                    hasattr(loaded_optimizer, 'opt_state') and
                    len(loaded_optimizer.opt_state) >= 2):

                    print(f"Original main step: {loaded_optimizer.step.value}")
                    print(f"Original Adam count: {loaded_optimizer.opt_state[0].count.value}")
                    print(f"Original Schedule count: {loaded_optimizer.opt_state[1].count.value}")
                    print("-" * 20)

                    # Create the new value we want to set
                    new_value = jnp.array(new_schedule_step, dtype=loaded_optimizer.step.value.dtype)

                    # 1. Modify the main step counter directly
                    loaded_optimizer.step.value = jnp.array(new_value, dtype=loaded_optimizer.step.value.dtype)

                    # 2. Modify the internal counts directly
                    #    Because the state objects are mutable, we can just assign the new value.
                    adam_state = loaded_optimizer.opt_state[0]
                    schedule_state = loaded_optimizer.opt_state[1]

                    adam_state.count.value = jnp.array(new_value, dtype=adam_state.count.value.dtype)
                    schedule_state.count.value = jnp.array(new_value, dtype=schedule_state.count.value.dtype)

                    print(f"✅ Modified main step: {loaded_optimizer.step.value}")
                    print(f"✅ Modified Adam count: {loaded_optimizer.opt_state[0].count.value}")
                    print(f"✅ Modified Schedule count: {loaded_optimizer.opt_state[1].count.value}")
                else:
                    print("❌ Unable to modify optimizer state: missing step or opt_state structure.")
                    print(f"Current optimizer state: {loaded_optimizer.opt_state}")
                    print("Using loaded optimizer as is, without modification.")

            optimizer = loaded_optimizer
            optimizer.model = model

            # IMPORTANT: Create a new mutable copy of the optimizer state
            # The loaded state is read-only, so we need to make it mutable
            optimizer_state = nnx.state(optimizer)
            model_state = nnx.state(model)
            
            # Create fresh mutable copies with better type checking
            def make_mutable_array(x):
                if hasattr(x, 'shape') and hasattr(x, 'dtype'):

                    # Convert to JAX array and ensure it's mutable
                    if isinstance(x, jnp.ndarray):
                        try: 
                            return jnp.array(np.array(x))  # Convert to numpy and back to JAX
                        except Exception as e:
                            print(f"Error converting to JAX array: {e}")
                            print(f"Dtype: {x.dtype}, Shape: {x.shape}, Value: {x}")
                            return x
                    else:
                        # Convert numpy or other array types to JAX arrays
                        return jnp.asarray(x)
                return x
            
            mutable_optimizer_state = jax.tree_util.tree_map(make_mutable_array, optimizer_state)
            mutable_model_state = jax.tree_util.tree_map(make_mutable_array, model_state)
            
            # Update with mutable state
            nnx.update(optimizer, mutable_optimizer_state)
            nnx.update(model, mutable_model_state)
            
            optimizer.model = model  # Ensure optimizer still points to the model

            # Ensure grad_accumulator is also mutable if it exists
            if grad_accumulator is not None:
                grad_accumulator = jax.tree_util.tree_map(lambda x: jnp.array(x) if hasattr(x, 'shape') else x, grad_accumulator)

            metrics_path = os.path.join(checkpoint_dir, "metrics.parquet")
            if os.path.exists(metrics_path):
                metrics_history = pd.read_parquet(metrics_path).to_dict('list')
            
            print(f"Resuming from Epoch {start_epoch}, Global Step {global_step}")
        else:
            print("No checkpoint found, starting training from scratch.")

    print(f"Starting training from epoch {start_epoch + 1} to {num_epochs}")

    # --- Main Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        model.train()
        metrics.reset()

        for batch_idx, batch in enumerate(train_dataloader):
            if resume_from_checkpoint and epoch == start_epoch and global_step > 0:
                if batch_idx < (global_step % len(train_dataloader)):
                    continue
            
            # SHARD: Move the batch from the host to the devices, splitting it.
            sharded_batch = jax.device_put(batch, data_sharding)

            # Pass the sharded batch to the JIT-compiled function.
            # JAX's compiler sees the sharded input and executes the function
            # in parallel, automatically handling gradient averaging.
            loss, new_grads_with_state = train_step(model, metrics, sharded_batch)

            # Unpack State objects to get raw JAX arrays
            new_grads = jax.tree_util.tree_map(_unpack_leaf, new_grads_with_state)

            # Initialize gradient accumulator on first batch
            if grad_accumulator is None:
                grad_accumulator = new_grads
                print("Gradient accumulator initialized with first batch gradients.")
            else:
                # Accumulate gradients
                def _add_or_keep(acc_leaf, new_leaf):
                    if new_leaf is None: return acc_leaf
                    if acc_leaf is None: return new_leaf
                    return acc_leaf + new_leaf

                grad_accumulator = jax.tree_util.tree_map(_add_or_keep, grad_accumulator, new_grads)

            global_step += 1

            # --- Optimizer Update ---
            if global_step % accumulation_steps == 0:
                def _scale(g): return g / accumulation_steps if g is not None else None
                avg_grads = jax.tree_util.tree_map(_scale, grad_accumulator)
                
                optimizer.update(avg_grads)
                
                # Reset accumulator to zeros with same structure
                def _create_zero_leaf(leaf):
                    if hasattr(leaf, 'shape') and hasattr(leaf, 'dtype'):
                        return jnp.zeros_like(leaf)
                    return None
                grad_accumulator = jax.tree_util.tree_map(_create_zero_leaf, grad_accumulator)
            
            # --- Metrics Logging & Saving ---
            if not metrics_history:
                # Get metric names from the MultiMetric object
                metric_names = list(metrics.__dict__.keys())
                for metric_name in metric_names:
                    metrics_history[f"train_{metric_name}"] = []
                    if test_dataloader:
                        metrics_history[f"test_{metric_name}"] = []
                metrics_history["epoch"], metrics_history["global_step"] = [], []
            
            if global_step % log_train_metrics_every_steps == 0:
                train_metrics = metrics.compute()
                metrics.reset()
                for name, val in train_metrics.items(): metrics_history[f"train_{name}"].append(float(val))
                if test_dataloader:
                    for name in train_metrics.keys(): metrics_history[f"test_{name}"].append(float('nan'))
                metrics_history["epoch"].append(epoch + 1)
                metrics_history["global_step"].append(global_step)
                train_metrics_str = ", ".join([f"{k.title()}: {v:.4f}" for k, v in train_metrics.items()])
                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                opt_step = optimizer.step.value if hasattr(optimizer, 'step') else 'N/A'
                print(f"{datetime_str} | Step {global_step:<7} | Epoch {epoch + 1:<4} | Train {train_metrics_str} | OPT_STEP: {opt_step}")

            if test_dataloader and eval_every_steps and (global_step % eval_every_steps == 0):
                model.eval()
                eval_metrics_computer = nnx.MultiMetric(
                    loss=nnx.metrics.Average('loss'),
                )
                for eval_batch in test_dataloader: 
                    # SHARD: Also shard the evaluation batch consistently
                    sharded_eval_batch = jax.device_put(eval_batch, data_sharding)
                    eval_step(model, eval_metrics_computer, sharded_eval_batch)
                test_metrics = eval_metrics_computer.compute()
                for name, val in test_metrics.items(): metrics_history[f"test_{name}"].append(float(val))
                for name in test_metrics.keys(): metrics_history[f"train_{name}"].append(float('nan'))
                metrics_history["epoch"].append(epoch + 1)
                metrics_history["global_step"].append(global_step)
                test_metrics_str = ", ".join([f"{k.title()}: {v:.4f}" for k, v in test_metrics.items()])
                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{datetime_str} | ** EVAL at Step {global_step:<7} | Epoch {epoch + 1:<4} | Test {test_metrics_str} **")
                model.train()

            # --- Checkpoint Saving ---
            if save_every_steps and (global_step % save_every_steps == 0):
                ckpt_name = f"epoch_{epoch:04d}_step_{global_step:08d}"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                _save_checkpoint_bundle(ckpt_path, model, optimizer, grad_accumulator, epoch, global_step)
                pd.DataFrame(dict([(k, pd.Series(v)) for k,v in metrics_history.items()])).to_parquet(os.path.join(checkpoint_dir, "metrics.parquet"), index=False)
        
        # --- End of Epoch Logic ---
        if (epoch + 1) % save_every_epochs == 0 or (epoch + 1) == num_epochs:
            def _is_leaf_dirty(leaf): return jnp.any(leaf != 0) if leaf is not None else False
            is_dirty = any(jax.tree_util.tree_leaves(jax.tree_util.tree_map(_is_leaf_dirty, grad_accumulator)))

            if is_dirty:
                print("Applying leftover gradients before end-of-epoch save...")
                steps_in_last_pile = global_step % accumulation_steps or accumulation_steps
                def _scale(g): return g / steps_in_last_pile if g is not None else g
                avg_grads = jax.tree_util.tree_map(_scale, grad_accumulator)
                optimizer.update(avg_grads)
                grad_accumulator = jax.tree_util.tree_map(_create_zero_leaf, grad_accumulator)

            ckpt_name = f"epoch_{epoch + 1:04d}_step_{global_step:08d}"
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            _save_checkpoint_bundle(ckpt_path, model, optimizer, grad_accumulator, epoch + 1, global_step)
            pd.DataFrame(dict([(k, pd.Series(v)) for k, v in metrics_history.items()])).to_parquet(os.path.join(checkpoint_dir, "metrics.parquet"), index=False)

    print("Training completed!")
    return model, metrics_history