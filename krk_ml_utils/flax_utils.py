# prompt: Write a function for saving a model and for loading model state.
# The save function should take a model and a string for the path for the safetensors file. The load function should take as input a string for the path for the safetensors file. It should update the model weights in place with the tensors in the file and then return the model to the calling function

import jax
import flax.nnx as nnx
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file
from jax._src.prng import PRNGKeyArray  # For type checking
import jax.random  # For key_data and wrap_key_data
from krk_ml_utils.models import MLPRegressor_v2, EnsembleModel, MLPRegressor_v3
import os
from krk_ml_utils import flax_utils
import pandas as pd

def save_model(model: nnx.Module, path: str):
    """Saves the complete state of an NNX model to a safetensors file.

    Args:
        model: The NNX model instance to save.
        path: The file path (including filename and .safetensors extension)
              where the model state will be saved.
    """
    # Split the model to get the graph definition and the state.
    # The state contains all variables (params, batch_stats, etc.).
    _, state = nnx.split(model)

    # Convert the state object to a dictionary.
    state_dict = state.to_pure_dict()

    # Flatten the dictionary and join keys with '.' for safetensors compatibility.
    # Safetensors expects a flat dictionary with string keys.
    flat_state = flatten_dict(state_dict)
    flat_state_joined = {".".join(map(str, k)): v for k, v in flat_state.items()}

    # Process PRNGKeyArrays to save their underlying data
    processed_flat_state_to_save = {}
    for k, v in flat_state_joined.items():
        if isinstance(v, PRNGKeyArray):
            processed_flat_state_to_save[k] = jax.random.key_data(v)
        else:
            processed_flat_state_to_save[k] = v

    # Save the processed flattened state to the safetensors file.
    save_file(processed_flat_state_to_save, path)
    print(f"NNX Model state successfully saved to {path}")


def load_model_state(model: nnx.Module, path: str) -> nnx.Module:
    """Loads model state from a safetensors file and returns an updated model.

    Args:
        model: The NNX model instance (used for its graph definition and type information).
        path: The file path (including filename and .safetensors extension)
              from which the model state will be loaded.

    Returns:
        A new NNX model instance updated with the loaded state.
    """
    # Load the flat dictionary from the safetensors file.
    loaded_flat_state_from_file = load_file(path)
    print(f"NNX Model state successfully loaded from {path}")

    # Get the graph definition and current state structure from the provided model instance.
    # The graph_def defines the model structure.
    # current_model_state_obj helps identify types like PRNGKeyArray for correct reconstruction.
    graph_def, current_model_state_obj = nnx.split(model)
    current_model_state_dict_flat = flatten_dict(current_model_state_obj.to_pure_dict())
    # Convert keys of current_model_state_dict_flat to the string format "path.to.key"
    current_model_flat_state_typed = {
        ".".join(map(str, k_tuple)): v
        for k_tuple, v in current_model_state_dict_flat.items()
    }

    reconstructed_values_for_unflatten = {}
    for k_str, loaded_jnp_array in loaded_flat_state_from_file.items():
        # Check if the current model expects a PRNGKeyArray at this path
        if k_str in current_model_flat_state_typed and isinstance(
            current_model_flat_state_typed[k_str], PRNGKeyArray
        ):
            # loaded_jnp_array is the key_data (a JAX uint32 array). Wrap it back.
            reconstructed_values_for_unflatten[k_str] = jax.random.wrap_key_data(
                loaded_jnp_array
            )
        else:
            reconstructed_values_for_unflatten[k_str] = loaded_jnp_array

    # Unflatten the dictionary to restore the nested structure.
    unflattened_state_dict = unflatten_dict(
        {tuple(k.split(".")): v for k, v in reconstructed_values_for_unflatten.items()}
    )

    # Convert the plain dictionary back into an nnx.State object.
    loaded_nnx_state = nnx.State(unflattened_state_dict)

    # Create a new model instance by merging the original graph_def
    # (from the 'model' argument) with the newly loaded state.
    updated_model = nnx.merge(graph_def, loaded_nnx_state)

    return updated_model

def load_best_test_loss_state(model: nnx.Module, save_dir: str) -> nnx.Module:
    """
    Loads the model state with the best test loss from a directory containing checkpoints.
    Args:
        model: The NNX model instance to update.
        save_dir: The directory where checkpoints and metrics are stored.
    Returns:
        The updated NNX model instance with the best test loss state loaded.

    """
    
    # Get metrics
    metrics = get_epochs_and_losses(save_dir)
    # Get checkpoints
    checkpoints = get_model_checkpoints(save_dir)
    # Make list of checkpoints with metric values 
    checkpoints_with_metrics = []
    for epoch, file_name in checkpoints:
        for metric_epoch, metric_value in metrics:
            if epoch == metric_epoch:
                checkpoints_with_metrics.append((epoch, file_name, metric_value))
                break
    # Sort by lowest metric value 
    checkpoints_with_metrics.sort(key=lambda x: x[2])  # Sort by metric value (3rd element)
    # get Absolute file name of the checkpoint with lowest metric value
    
    checkpoint_filename = os.path.join(save_dir, checkpoints_with_metrics[0][1])
    
    model = load_model_state(model, checkpoint_filename)
    print(f"Loaded model state from {checkpoint_filename} with best test loss: {checkpoints_with_metrics[0][2]}")
    
    return model

def compare_models(model1: nnx.Module, model2: nnx.Module) -> bool:
    """Checks if all weights and parameters of two NNX models are the same.

    Args:
        model1: The first NNX model instance.
        model2: The second NNX model instance.

    Returns:
        True if all parameters are the same, False otherwise.
    """
    # Split both models to get their states
    _, state1 = nnx.split(model1)
    _, state2 = nnx.split(model2)

    # Convert states to dictionaries and flatten
    flat_state1 = flatten_dict(state1.to_pure_dict())
    flat_state2 = flatten_dict(state2.to_pure_dict())

    # Check if the keys (parameter names and paths) are the same
    if set(flat_state1.keys()) != set(flat_state2.keys()):
        print("Model structures (keys) do not match.")
        return False

    # Compare the values (the actual JAX arrays)
    for key in flat_state1.keys():
        if not jax.numpy.array_equal(flat_state1[key], flat_state2[key]):
            print(f"Parameters do not match for key: {key}")
            return False

    print("All model parameters match.")
    return True


def get_models(
    n: int = 1,
    num_features=12,
    layer_sizes=(32, 16, 16, 8, 8, 4, 2, 1),
    num_categorical=1,
    embed_dim=3,
    dropout_rate=0.5,
    rngs=nnx.Rngs(42),
):
    """
    Creates a list of initialized MLPRegressor_v2 models.

    This function generates multiple instances of the MLPRegressor_v2 model,
    each with a unique random seed for its initial parameters. This is
    useful for building model ensembles.

    Args:
        n: The number of models to create.
        num_features: The number of input features for the model.
        layer_sizes: A tuple defining the size of each layer.
        num_categorical: The number of categorical features.
        embed_dim: The embedding dimension for categorical features.
        dropout_rate: The dropout rate for regularization.
        rngs: The JAX random number generator state.

    Returns:
        A list of initialized MLPRegressor_v2 models.
    """
    key = rngs.default()

    models = []
    random_integers = jax.random.randint(
        key,
        shape=(n,),
        minval=1,
        maxval=999,
    )
    for i in random_integers:

        model = MLPRegressor_v2(
            num_features=num_features,  # Number of input features
            layer_sizes=layer_sizes,  # Output layer size is 1 for regression
            num_categorical=num_categorical,  # Number of categorical features
            embed_dim=embed_dim,  # Embedding dimension for categorical features
            dropout_rate=dropout_rate,  # Dropout rate for regularization
            rngs=nnx.Rngs(int(i)),  # Random number generator state
        )
        models.append(model)

    return models


def get_models_v3(
    n: int = 1,
    num_features=12,
    layer_sizes=(32, 16, 16, 8, 8, 4, 2, 1),
    categorical_configs=None,
    dropout_rate=0.5,
    rngs=nnx.Rngs(42),
):
    """
    Creates a list of initialized MLPRegressor_v2 models.

    This function generates multiple instances of the MLPRegressor_v2 model,
    each with a unique random seed for its initial parameters. This is
    useful for building model ensembles.

    Args:
        n: The number of models to create.
        num_features: The number of input features for the model.
        layer_sizes: A tuple defining the size of each layer.
        num_categorical: The number of categorical features.
        embed_dim: The embedding dimension for categorical features.
        dropout_rate: The dropout rate for regularization.
        rngs: The JAX random number generator state.

    Returns:
        A list of initialized MLPRegressor_v2 models.
    """
    key = rngs.default()

    models = []
    random_integers = jax.random.randint(
        key,
        shape=(n,),
        minval=1,
        maxval=999,
    )
    for i in random_integers:

        model = MLPRegressor_v3(
            num_features=num_features,  # Adjusted for the number of features in the dataset
            layer_sizes=layer_sizes,  # Example layer sizes, can be adjusted
            categorical_configs=categorical_configs,  # Assuming no categorical features in this dataset
            dropout_rate=dropout_rate,  # Example dropout rate
            rngs=nnx.Rngs(int(i))  # Initialize RNGs with a fixed key
        )

        models.append(model)

    return models

# Function constructs an nnx Model out of the ensemble of models
# It takes a parent directory, and a directory name pattern for ensembles
# It makes al ist of all directories that match the pattern,
# For each directory, it looks for the latest checkpoint and loads the model and appends the model to the list of models
def load_ensemble_models(
    parent_dir,
    model_factory,
    ensemble_pattern="checkpoint_model_"
):

    # Get all directories that match the ensemble pattern in the parent dir
    ensemble_dirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if d.startswith(ensemble_pattern)
    ]
    num_models = len(ensemble_dirs)
    # print(f"Found {num_models} ensemble directories matching pattern '{ensemble_pattern}' in '{parent_dir}'")

    # Generate models using the model factory
    models = model_factory(n=num_models, rngs=nnx.Rngs(42))
    for i, ensemble_dir in enumerate(ensemble_dirs):
        model = flax_utils.load_best_test_loss_state(models[i], ensemble_dir)
        models[i] = model

    # Create an ensemble model that averages predictions from all models
    ensemble_model = EnsembleModel(models=models)

    return ensemble_model


def prune_checkpoints(
    save_dir: str,
    max_checkpoints: int = 5,
    keep_best: bool = True,
):
    """
    Prunes the checkpoints in the save directory to keep only the latest `max_checkpoints` files.
    If `keep_best` is True, it keeps the checkpoint with the best test loss.

    Args:
        save_dir: The directory where checkpoints are stored.
        max_checkpoints: The maximum number of checkpoints to keep.
        keep_best: Whether to keep the checkpoint with the best test loss.

    Returns:
        None
    """
    checkpoints = get_model_checkpoints(save_dir)
    
    if not checkpoints:
        print("No checkpoints found to prune.")
        return

    # If we want to keep the best checkpoint, we need to find it
    if keep_best:
        epochs_and_losses = get_epochs_and_losses(save_dir)
        best_checkpoint = min(epochs_and_losses, key=lambda x: x[1])  # Get the one with lowest loss
        best_epoch = best_checkpoint[0]
        best_file_name = f"epoch_{best_epoch:04d}_*.safetensors"
        print(f"Keeping best checkpoint: {best_file_name}")

    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: x[0], reverse=True)

    # Keep only the latest `max_checkpoints` or the best one if specified
    checkpoints_to_keep = set()
    for i, (epoch, file_name) in enumerate(checkpoints):
        if i < max_checkpoints or (keep_best and file_name.startswith(f"epoch_{best_epoch:04d}")):
            checkpoints_to_keep.add(file_name)

    # Remove old checkpoints
    for file_name in os.listdir(save_dir):
        if file_name.startswith("epoch_") and file_name.endswith(".safetensors"):
            if file_name not in checkpoints_to_keep:
                os.remove(os.path.join(save_dir, file_name))
                print(f"Removed old checkpoint: {file_name}")


def get_model_checkpoints(save_dir: str) ->  list[tuple[int, str]]:
    """    Returns a list of tuples containing epoch number and file name for each checkpoint in the save directory.
    """
    #Example file name for epoch 150: epoch_0150_20250625_200447.safetensors
    
    checkpoints = []
    for file_name in os.listdir(save_dir):
        if file_name.startswith("epoch_") and file_name.endswith(".safetensors"):
            # Extract the epoch number from the file name
            epoch_str = file_name.split("_")[1]
            epoch = int(epoch_str)  # Convert to integer
            checkpoints.append((epoch, file_name))
    
    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def get_epochs_and_losses(save_dir: str) -> list[tuple]:
    pass

    # Read metrics parquet
    metrics_file = os.path.join(save_dir, "metrics.parquet")
    df = pd.read_parquet(metrics_file)
    
    # Extract epochs and losses
    epochs = df['epoch'].tolist()
    losses = df['test_loss'].tolist()
    
    # Return as a list of tuples
    return list(zip(epochs, losses))