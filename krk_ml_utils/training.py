import jax
import optax
import os
import cloudpickle
from datetime import datetime
from typing import Callable, Tuple, Optional, Dict
import jax.numpy as jnp
import pandas as pd
from flax import nnx
from .flax_utils import save_model, load_model_state
from krk_ml_utils.datasets import JaxDataLoader, NumpyDataset
from krk_ml_utils.flax_utils import get_models


def save_optimizer_state(optimizer: nnx.Optimizer, path: str):
    """Saves the optimizer state to a file using cloudpickle."""

    with open(path, "wb") as f:
        cloudpickle.dump(optimizer, f)
    print(f"Optimizer state saved to {path}")


def load_optimizer_state(path: str) -> nnx.Optimizer:
    """Loads the optimizer state from a file using cloudpickle."""

    with open(path, "rb") as f:
        optimizer = cloudpickle.load(f)
    print(f"Optimizer state loaded from {path}")
    return optimizer


def shuffle_dataset(dataset, key):
    perm = jax.random.permutation(key, len(dataset))
    return dataset[perm]


def train_model(
    model_fn: Callable,
    loss_fn: Callable,
    params: Optional[jax.Array] = None,
    opt_state: Optional[optax.OptState] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    num_epochs: int = 100,
    train_ds=None,
    val_ds=None,
    checkpoint_dir: str = "checkpoints",
    resume_from_checkpoint: bool = False,
) -> Tuple[jax.Array, optax.OptState]:
    """
    Full-featured training function with checkpoint resuming
    """

    # Create or validate checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint loading logic
    start_epoch = 0
    if resume_from_checkpoint:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            params, opt_state, start_epoch = load_checkpoint(latest_ckpt)
            print(f"Resuming training from epoch {start_epoch} at {latest_ckpt}")
        elif params is None or opt_state is None or optimizer is None:
            raise ValueError(
                "No checkpoint found and missing required initialization parameters"
            )

    # Training infrastructure
    @jax.jit
    def train_step(state, batch):
        x, y = batch
        x = jax.numpy.array(x)
        y = jax.numpy.array(y)
        params, opt_state = state
        grads = jax.grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), compute_metrics(
            new_params, batch, model_fn, loss_fn
        )

    jit_train_step = jax.jit(train_step)

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        epoch_loss = 0
        state = (params, opt_state)
        for batch in train_ds:
            # TODO: Clean this up for performance reasons
            # The batch comes across as a tuple of torch tensors
            # It needs to be jax arrays
            x, y = batch
            x = jax.numpy.array(x)
            y = jax.numpy.array(y)
            batch = (x, y)
            state, metrics = jit_train_step(state, batch)
            epoch_loss += metrics["loss"]

        # Update tracking
        params, opt_state = state

        # Validation phase
        val_metrics = evaluate(model_fn, params, loss_fn, val_ds)

        # Checkpointing
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"epoch_{epoch+1}_{timestamp}.ckpt"
        )

        save_checkpoint(
            path=checkpoint_path,
            params=params,
            opt_state=opt_state,
            epoch=epoch + 1,
            train_loss=epoch_loss / len(train_ds),
            val_loss=val_metrics["loss"],
        )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f" t-loss: {epoch_loss/len(train_ds):.4f}")
        print(f" v-loss: {val_metrics['loss']:.4f}")
        print(f" ckpt: {checkpoint_path}")

    return params, opt_state


# Utility functions -----------------------------------------------------------


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds most recent checkpoint in directory"""
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None

    # Extract epoch numbers and find max
    epoch_numbers = []
    for f in ckpts:
        try:
            epoch = int(f.split("_")[1])
            epoch_numbers.append((epoch, f))
        except (IndexError, ValueError):
            continue

    if not epoch_numbers:
        return None

    # Return path to latest checkpoint
    latest = max(epoch_numbers, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest[1])


def save_checkpoint(path: str, **kwargs):
    """Save training state to checkpoint"""
    with open(path, "wb") as f:
        cloudpickle.dump(kwargs, f)


def load_checkpoint(path: str) -> Tuple[jax.Array, optax.OptState, int]:
    """Load training state from checkpoint"""
    with open(path, "rb") as f:
        ckpt = cloudpickle.load(f)
    return ckpt["params"], ckpt["opt_state"], ckpt["epoch"]


def evaluate(model_fn, params, loss_fn, dataset):
    """Evaluation function"""
    total_loss = 0
    for batch in dataset:
        x, y = batch
        x = jax.numpy.array(x)
        y = jax.numpy.array(y)
        # preds = model_fn(params, x)
        total_loss += loss_fn(params, x, y)
    return {"loss": total_loss / len(dataset)}


def compute_metrics(params, batch, model_fn, loss_fn):
    """Metric computation (example implementation)"""
    x, y = batch
    x = jax.numpy.array(x)
    y = jax.numpy.array(y)
    # preds = model_fn(params, x)
    return {"loss": loss_fn(params, x, y)}


def train_flax_model(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    loss_fn: Callable,
    train_dataloader: JaxDataLoader,
    test_dataloader: Optional[JaxDataLoader] = None,
    eval_fn: Optional[Callable] = None,
    num_epochs: int = 10,
    checkpoint_dir: str = "checkpoints",
    save_every: int = 1,  # Save checkpoint every N epochs
    metrics_save_every: int = 1,  # Save metrics every N epochs
    resume_from_checkpoint: bool = False,
    early_stopping_patience: Optional[int] = None,  # Optional early stopping patience
    loss_decimals: int = 4,  # Optional fudge factor for loss scaling
) -> Tuple[nnx.Module, Dict[str, list]]:
    """
    Train a Flax NNX model using custom loss, evaluation functions, and metrics.

    Args:
        model: The Flax NNX model to train
        optimizer: The Flax NNX optimizer (nnx.Optimizer instance)
        metrics: The Flax NNX MultiMetric instance for tracking training metrics
        loss_fn: Loss function that takes (model, batch) and returns (loss, logits)
                 Should match signature: loss_fn(model: nnx.Module, batch) -> Tuple[loss, logits]
        train_dataloader: JaxDataLoader for training data
        test_dataloader: Optional JaxDataLoader for test/validation data
        eval_fn: Optional evaluation function. If None, uses the same loss_fn for evaluation.
                 Should match signature: eval_fn(model: nnx.Module, batch) -> Tuple[loss, logits]
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints and metrics
        save_every: Save checkpoint every N epochs
        metrics_save_every: Save metrics to parquet every N epochs
        resume_from_checkpoint: Whether to resume from latest checkpoint

    Returns:
        Tuple of (trained_model, metrics_history)
    """

    # Use loss_fn for evaluation if no specific eval_fn is provided
    if eval_fn is None:
        eval_fn = loss_fn

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get the metric names from the MultiMetric for dynamic history tracking
    # We'll discover the actual metric names after the first computation
    metrics_history = {"epoch": [], "timestamp": []}

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume_from_checkpoint:
        latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            model = load_model_state(model, latest_checkpoint)
            start_epoch = _extract_epoch_from_checkpoint(latest_checkpoint)

            print(f"Loading metrics history from {latest_checkpoint}")
            # Load the saved metrics history if available
            metrics_history_df = load_metrics_history(checkpoint_dir)
            metrics_hist_temp = {}

            # Create a key for each column and make the value a list of the values for the column
            for col in metrics_history_df.columns:
                metrics_hist_temp[col] = metrics_history_df[col].tolist()

            metrics_history = metrics_hist_temp

            optimizer_path = os.path.join(checkpoint_dir, "optimizer_state.pkl")
            print(f"Loading optimizer state from {optimizer_path}")
            if os.path.exists(optimizer_path):
                optimizer = load_optimizer_state(optimizer_path)
                # Reset the optimizer model reference
                optimizer.model = model  # Ensure optimizer references the current model
                print("Optimizer state loaded successfully")
            else:
                print("No optimizer state found, using the provided optimizer")

            # if metrics_history_df is not None:
            #    metrics_history = metrics_history_df.to_dict(orient='list')
            print(
                f"Resumed from checkpoint: {latest_checkpoint}, starting at epoch {start_epoch}"
            )
        else:
            print("No checkpoint found, starting from scratch")

    # Define training step using the provided loss_fn
    @nnx.jit
    def train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
    ):
        """Train for a single step using the provided loss function."""
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)

        # Handle different batch formats for metrics
        if isinstance(batch, dict):
            labels = batch["label"]
        else:
            _, labels = batch

        metrics.update(loss=loss, logits=logits, labels=labels)
        optimizer.update(grads)

    # Define evaluation step using the provided eval_fn
    @nnx.jit
    def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
        """Evaluate for a single step using the provided evaluation function."""
        loss, logits = eval_fn(model, batch)

        # Handle different batch formats for metrics
        if isinstance(batch, dict):
            labels = batch["label"]
        else:
            _, labels = batch

        metrics.update(loss=loss, logits=logits, labels=labels)

    print(f"Starting training from epoch {start_epoch + 1} to {num_epochs}")

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        metrics.reset()  # Reset metrics at the start of each epoch
        # Training phase
        model.train()  # Set to training mode
        train_batches = 0

        for batch in train_dataloader:
            train_step(model, optimizer, metrics, batch)
            train_batches += 1

        # Compute training metrics
        train_metrics = metrics.compute()
        metrics.reset()

        # Evaluation phase
        test_metrics = {}
        if test_dataloader is not None:
            model.eval()  # Set to evaluation mode

            for batch in test_dataloader:
                eval_step(model, metrics, batch)

            test_metrics = metrics.compute()
            metrics.reset()

        # Initialize metrics history if this is the first epoch
        if epoch == 0:
            for metric_name in train_metrics.keys():
                metrics_history[f"train_{metric_name}"] = []
                if test_dataloader is not None:
                    metrics_history[f"test_{metric_name}"] = []

        # Record metrics
        current_time = datetime.utcnow().isoformat()
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["timestamp"].append(current_time)

        # Add training metrics to history
        for metric_name, value in train_metrics.items():
            metrics_history[f"train_{metric_name}"].append(float(value))

        # Add test metrics to history
        for metric_name, value in test_metrics.items():
            metrics_history[f"test_{metric_name}"].append(float(value))

        # Print progress dynamically based on available metrics
        train_metrics_str = ", ".join(
            [f"Train {k.title()}: {v:.4f}" for k, v in train_metrics.items()]
        )
        print(f"  {train_metrics_str}")

        if test_dataloader is not None and test_metrics:
            test_metrics_str = ", ".join(
                [f"Test {k.title()}: {v:.4f}" for k, v in test_metrics.items()]
            )
            print(f"  {test_metrics_str}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_filename = f"epoch_{epoch + 1:04d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.safetensors"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            save_model(model, checkpoint_path)
            save_optimizer_state(
                optimizer, os.path.join(checkpoint_dir, "optimizer_state.pkl")
            )
            print(f"  Saved checkpoint: {checkpoint_filename}")

        # Save metrics to parquet

        if (epoch + 1) % metrics_save_every == 0 or epoch == num_epochs - 1:
            metrics_df = pd.DataFrame(metrics_history)
            metrics_filename = "metrics.parquet"
            metrics_path = os.path.join(checkpoint_dir, metrics_filename)
            metrics_df.to_parquet(metrics_path, index=False)
            print(f"  Saved metrics: {metrics_path}")

        # Early stopping logic
        if early_stopping_patience is not None:
            if epoch >= early_stopping_patience:
                # Check if we have test data and enough history
                if (
                    test_dataloader is not None
                    and "test_loss" in metrics_history
                    and len(metrics_history["test_loss"]) > early_stopping_patience
                ):
                    recent_losses = metrics_history["test_loss"][
                        -early_stopping_patience:
                    ]
                    # Check all losses within decimal precision
                    recent_losses = [
                        round(loss, loss_decimals) for loss in recent_losses
                    ]
                    # Check if all recent losses are greater than or equal to the first one (no improvement)
                    if all(loss >= recent_losses[0] for loss in recent_losses):
                        print(
                            f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss."
                        )
                        print(f"Recent losses: {recent_losses}")
                        checkpoint_filename = f"epoch_{epoch + 1:04d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.safetensors"
                        checkpoint_path = os.path.join(
                            checkpoint_dir, checkpoint_filename
                        )
                        save_model(model, checkpoint_path)
                        save_optimizer_state(
                            optimizer,
                            os.path.join(checkpoint_dir, "optimizer_state.pkl"),
                        )
                        print(f"  Saved checkpoint: {checkpoint_filename}")
                        break

    print("Training completed!")
    return model, metrics_history


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")]
    if not checkpoints:
        return None

    # Sort by epoch number (assuming format: epoch_XXXX_timestamp.safetensors)
    def extract_epoch(filename):
        try:
            parts = filename.split("_")
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    latest_checkpoint = max(checkpoints, key=extract_epoch)
    return os.path.join(checkpoint_dir, latest_checkpoint)


def _extract_epoch_from_checkpoint(checkpoint_path: str) -> int:
    """Extract epoch number from checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    try:
        parts = filename.split("_")
        return int(parts[1])
    except (IndexError, ValueError):
        return 0


def load_metrics_history(checkpoint_dir: str) -> Optional[pd.DataFrame]:
    """Load the latest metrics history from parquet files."""
    if not os.path.exists(checkpoint_dir):
        return None

    metrics_path = os.path.join(checkpoint_dir, "metrics.parquet")

    return pd.read_parquet(metrics_path)


# Example loss and evaluation functions
def classification_loss_fn(model: nnx.Module, batch):
    """Example classification loss function matching the MNIST notebook style."""
    # Handle both dict-style batches (like MNIST) and tuple batches
    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch["label"]
    else:
        images, labels = batch

    logits = model(images)

    # Handle different label formats
    if labels.ndim == 1:  # Integer labels
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
    else:  # One-hot labels
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

    return loss, logits


def regression_loss_fn(model: nnx.Module, batch):
    """Example regression loss function."""
    if isinstance(batch, dict):
        features = batch["features"]
        targets = batch["targets"]
    else:
        features, targets = batch

    predictions = model(features)
    loss = jnp.mean((predictions - targets) ** 2)  # MSE loss

    return loss, predictions


# Example usage functions with different metrics configurations
def create_classification_example():
    """Example for classification with accuracy and loss metrics."""
    from functools import partial

    # Define CNN model (same as MNIST notebook)
    class CNN(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
            self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
            self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
            self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
            self.linear2 = nnx.Linear(256, 10, rngs=rngs)

        def __call__(self, x):
            x = self.avg_pool(nnx.relu(self.conv1(x)))
            x = self.avg_pool(nnx.relu(self.conv2(x)))
            x = x.reshape(x.shape[0], -1)  # flatten
            x = nnx.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    # Create model
    model = CNN(rngs=nnx.Rngs(0))

    # Create optimizer (same as MNIST notebook)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.005, b2=0.9))

    # Create metrics - same as MNIST notebook
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    return model, optimizer, metrics


def create_regression_example():
    """Example for regression with custom metrics."""

    # Simple regression model
    class RegressionModel(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(10, 64, rngs=rngs)
            self.linear2 = nnx.Linear(64, 32, rngs=rngs)
            self.linear3 = nnx.Linear(32, 1, rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.linear1(x))
            x = nnx.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    model = RegressionModel(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001))

    # Regression metrics
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        mae=nnx.metrics.Average("mae"),  # Mean Absolute Error
    )

    return model, optimizer, metrics


def create_advanced_classification_example():
    """Example with more comprehensive metrics for classification."""
    from functools import partial

    class CNN(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
            self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
            self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
            self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
            self.linear2 = nnx.Linear(256, 10, rngs=rngs)

        def __call__(self, x):
            x = self.avg_pool(nnx.relu(self.conv1(x)))
            x = self.avg_pool(nnx.relu(self.conv2(x)))
            x = x.reshape(x.shape[0], -1)  # flatten
            x = nnx.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    model = CNN(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.005, b2=0.9))

    # Advanced metrics with multiple tracking
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
        top_k_accuracy=nnx.metrics.TopKAccuracy(k=3),
    )

    return model, optimizer, metrics


# Example usage:
def example_usage():
    """Example of how to use the updated train_flax_model function."""

    # For basic classification (MNIST-style)
    model, optimizer, metrics = create_classification_example()

    # Train the model
    # trained_model, history = train_flax_model(
    #     model=model,
    #     optimizer=optimizer,
    #     metrics=metrics,
    #     loss_fn=classification_loss_fn,
    #     train_dataloader=train_loader,
    #     test_dataloader=test_loader,
    #     num_epochs=10,
    #     checkpoint_dir="./checkpoints"
    # )

    # For regression
    # reg_model, reg_optimizer, reg_metrics = create_regression_example()
    # trained_reg_model, reg_history = train_flax_model(
    #     model=reg_model,
    #     optimizer=reg_optimizer,
    #     metrics=reg_metrics,
    #     loss_fn=regression_loss_fn,
    #     train_dataloader=train_loader,
    #     test_dataloader=test_loader,
    #     num_epochs=100,
    #     checkpoint_dir="./regression_checkpoints"
    # )

    return model, optimizer, metrics


def train_bag(
    model_factory,
    model_seed,
    dataset_seed,
    num_models,
    experiment_dir,
    train_npz,
    test_npz,
    loss_fn,
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=2000,
    save_every=150,
    metrics_save_every=1,
    early_stopping_patience=150,
    loss_decimals=7,
):


    # If ensemble directory exists, check hwo many models are already trained 
    num_existing_models = 0
    ensemble_dir = os.path.join(experiment_dir, "ensemble")
    if os.path.exists(ensemble_dir):
        existing_models = [
            f for f in os.listdir(ensemble_dir) if f.startswith("checkpoint_model_")
        ]
        num_existing_models = len(existing_models)
        print(f"Found {num_existing_models} existing models in ensemble directory.")
        
    
    num_models_input = num_models 
    num_models = num_models - num_existing_models
    if num_models <= 0:
        print("No new models to train. Exiting.")
        return
    # Generate randomly initialized models using factory function and seed
    models = model_factory(n=num_models, rngs=nnx.Rngs(model_seed))

    # Get a dataloder for the test set
    test_ds = NumpyDataset(test_npz)
    test_loader = JaxDataLoader(test_ds, batch_size=batch_size, shuffle=False)
     

    # Make random numbers for the seeds of the training loaders
    rng = nnx.Rngs(dataset_seed)
    key = rng.default()
    random_integers = jax.random.randint(key, shape=(num_models,), minval=1, maxval=999)
    train_loaders = []
    # Create data loaders for each model with different seeds
    optimizers = []
    metrics_list = []
    checkpoint_dirs = []
    model_number = 0
    for i in random_integers:
        train_ds = NumpyDataset(train_npz, rngs=nnx.Rngs(int(i)))
        train_loader = JaxDataLoader(train_ds, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)

        # Define the loss function for training
        model = models[model_number]
        optimizer = nnx.Optimizer(
            model, optax.adamw(learning_rate=learning_rate, b2=0.9, weight_decay=1e-4)
        )  # Add explicit weight decay
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )
        optimizers.append(optimizer)
        metrics_list.append(metrics)
        current_ckpt = model_number+num_existing_models-1
        
        if current_ckpt < 0:
            current_ckpt = 0
        # Create a directory for each model's checkpoints
        checkpoint_dir = os.path.join(
            experiment_dir, "ensemble", f"checkpoint_model_{current_ckpt}"
        )
        checkpoint_dirs.append(checkpoint_dir)
        model_number += 1

    # Train each model in the ensemble

    num_experts = len(models)

    for i in range(num_experts):
        model = models[i]
        train_loader = train_loaders[i]
        optimizer = optimizers[i]
        metrics = metrics_list[i]
        checkpoint_dir = checkpoint_dirs[i]

        # Train the model
        train_flax_model(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            loss_fn=loss_fn,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            save_every=save_every,  # Save checkpoint every 2 epochs
            metrics_save_every=metrics_save_every,  # Save metrics every epoch
            resume_from_checkpoint=True,
            early_stopping_patience=early_stopping_patience,  # Early stopping patience
            loss_decimals=loss_decimals,  # Fudge factor for loss scaling
        )
