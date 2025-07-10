from flax import nnx  # Import the NNX API for Flax.
import optax  # Import Optax for optimizers.
import jax.numpy as jnp


def mse_loss_with_l1(model: nnx.Module, batch, l1_lambda: float = 1e-4):
    """Loss function with L1 regularization that excludes batch norm and embedding parameters."""
    images, labels = batch
    logits = model(images)

    # Convert labels to proper format if needed
    if labels.ndim > 1:
        labels = labels.squeeze()

    # Ensure logits have the same shape as labels
    if logits.ndim > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)

    # Calculate MSE loss
    mse_loss = optax.l2_loss(logits, labels)
    mse_loss = jnp.mean(mse_loss)

    # Calculate L1 regularization only on linear layer weights
    l1_loss = 0.0
    state = nnx.state(model)

    # Only regularize 'layers' -> layer_index -> 'kernel'
    if "layers" in state:
        for layer_idx, layer_params in state["layers"].items():
            if "kernel" in layer_params:
                kernel_param = layer_params["kernel"]
                if hasattr(kernel_param, "value"):
                    l1_loss += jnp.sum(jnp.abs(kernel_param.value))

    # Combine MSE loss with L1 regularization
    total_loss = mse_loss + l1_lambda * l1_loss

    return total_loss, logits


# Custom loss function for your data format - simplified version
def mse_loss(model: nnx.Module, batch):
    """Loss function that handles tuple batch format from JaxDataLoader."""
    images, labels = batch  # Unpack tuple format
    logits = model(images)

    # Convert labels to proper format if needed
    if labels.ndim > 1:
        labels = labels.squeeze()

    # Ensure logits have the same shape as labels
    # Your model outputs (batch_size, 1) but labels are (batch_size,)
    if logits.ndim > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)  # Remove the last dimension: (32, 1) -> (32,)

    # Calculate MSE loss
    loss = optax.l2_loss(logits, labels)
    loss = jnp.mean(loss)  # Average loss over the batch

    # Return loss and logits - the training function will calculate MAE automatically
    return loss, logits
