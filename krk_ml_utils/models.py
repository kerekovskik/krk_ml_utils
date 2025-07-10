from typing import Tuple
from flax import nnx
import jax.numpy as jnp


class MLPRegressor(nnx.Module):
    """
    Multi-Layer Perceptron for regression tasks with categorical feature embeddings.

    This model handles both numerical and categorical features by:
    1. Applying embeddings to categorical features
    2. Concatenating numerical features with categorical embeddings
    3. Passing the combined features through MLP layers
    4. Outputting a single regression value
    """

    def __init__(
        self,
        num_features: int,
        layer_sizes: Tuple[int, ...],
        num_categorical: int,
        embed_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the MLP Regressor.

        Args:
            num_features: Total number of input features (numerical + categorical)
            num_layers: Number of layers in the MLP
            layer_sizes: Tuple defining the layer dimensions
            num_categorical: Number of categorical features (rightmost columns)
            embed_dim: Dimension of embedding vectors for categorical features
            dropout_rate: Dropout rate for regularization
            rngs: Random number generator state
        """
        # Store configuration parameters
        self.num_features = num_features
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.num_categorical = num_categorical
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        # Calculate number of numerical features
        # Categorical features are at the rightmost columns
        self.num_numerical = num_features - num_categorical

        # Create embedding layers for each categorical feature
        # Each categorical feature gets its own embedding layer
        self.embeddings = []
        for i in range(num_categorical):
            # Note: vocab_size should be set based on your actual categorical data
            # For now, using a reasonable default of 100
            embed_layer = nnx.Embed(
                num_embeddings=num_categorical,  # Adjust based on your categorical feature cardinality
                features=embed_dim,
                rngs=rngs,
            )
            self.embeddings.append(embed_layer)

        # Calculate input dimension for first MLP layer
        # = numerical features + (categorical features * embedding dimension)
        first_layer_input_dim = self.num_numerical + (num_categorical * embed_dim)

        # Create MLP layers based on layer_sizes specification
        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                # First layer: input from combined numerical + embedded categorical features
                input_dim = first_layer_input_dim
                output_dim = layer_sizes[i]
            else:
                # Subsequent layers: input from previous layer
                input_dim = layer_sizes[i - 1]
                output_dim = layer_sizes[i]

            # Create linear layer
            layer = nnx.Linear(
                in_features=input_dim, out_features=output_dim, rngs=rngs
            )
            self.layers.append(layer)

        # Create dropout layer for regularization
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, training: bool = False):
        """
        Forward pass through the MLP.

        Args:
            x: Input array of shape (batch_size, num_features)
               Categorical features are in the rightmost num_categorical columns
            training: Whether model is in training mode (affects dropout)

        Returns:
            Regression predictions of shape (batch_size, 1)
        """
        # batch_size = x.shape[0]

        # Split input into numerical and categorical features
        if self.num_categorical > 0:
            # Numerical features are the leftmost columns
            numerical_features = x[
                :, : self.num_numerical
            ]  # Shape: (batch_size, num_numerical)

            # Categorical features are the rightmost columns
            categorical_features = x[:, self.num_numerical:]  # Shape: (batch_size, num_categorical)

            # Apply embeddings to each categorical feature
            embedded_categoricals = []
            for i in range(self.num_categorical):
                # Extract the i-th categorical feature column
                cat_feature = categorical_features[:, i].astype(
                    jnp.int32
                )  # Shape: (batch_size,)

                # Apply embedding layer to get embedding vectors
                embedded = self.embeddings[i](
                    cat_feature
                )  # Shape: (batch_size, embed_dim)
                embedded_categoricals.append(embedded)

            # Concatenate all categorical embeddings
            if embedded_categoricals:
                cat_embeddings = jnp.concatenate(
                    embedded_categoricals, axis=1
                )  # Shape: (batch_size, num_categorical * embed_dim)

                # Combine numerical features with categorical embeddings
                combined_features = jnp.concatenate(
                    [numerical_features, cat_embeddings], axis=1
                )
            else:
                combined_features = numerical_features
        else:
            # No categorical features, use all features as numerical
            combined_features = x

        # Pass through MLP layers
        output = combined_features

        for i, layer in enumerate(self.layers):
            # Apply linear transformation
            output = layer(output)

            # Apply ReLU activation to all layers except the last one
            # Last layer is linear for regression output
            if i < len(self.layers) - 1:
                output = nnx.relu(output)

                # Apply dropout during training for regularization
                if training:
                    output = self.dropout(output, deterministic=False)
                else:
                    output = self.dropout(output, deterministic=True)

        # Final output for regression (no activation on last layer)
        return output


class MLPRegressor_v2(nnx.Module):
    """
    Multi-Layer Perceptron for regression tasks with categorical feature embeddings.

    This model handles both numerical and categorical features by:
    1. Applying embeddings to categorical features
    2. Concatenating numerical features with categorical embeddings
    3. Passing the combined features through MLP layers
    4. Outputting a single regression value
    """

    def __init__(
        self,
        num_features: int,
        layer_sizes: Tuple[int, ...],
        num_categorical: int,
        embed_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the MLP Regressor.

        Args:
            num_features: Total number of input features (numerical + categorical)
            num_layers: Number of layers in the MLP
            layer_sizes: Tuple defining the layer dimensions
            num_categorical: Number of categorical features (rightmost columns)
            embed_dim: Dimension of embedding vectors for categorical features
            dropout_rate: Dropout rate for regularization
            rngs: Random number generator state
        """
        # Store configuration parameters
        self.num_features = num_features
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.num_categorical = num_categorical
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        # Calculate number of numerical features
        # Categorical features are at the rightmost columns
        self.num_numerical = num_features - num_categorical

        # Create embedding layers for each categorical feature
        # Each categorical feature gets its own embedding layer
        self.embeddings = []
        for i in range(num_categorical):
            # Note: vocab_size should be set based on your actual categorical data
            # For now, using a reasonable default of 100
            embed_layer = nnx.Embed(
                num_embeddings=num_categorical,  # Adjust based on your categorical feature cardinality
                features=embed_dim,
                rngs=rngs,
            )
            self.embeddings.append(embed_layer)

        # Calculate input dimension for first MLP layer
        # = numerical features + (categorical features * embedding dimension)
        first_layer_input_dim = self.num_numerical + (num_categorical * embed_dim)

        # Create MLP layers based on layer_sizes specification
        self.layers = []

        # List of batch normalization layers
        self.bns = []
        for i in range(self.num_layers):
            if i == 0:
                # First layer: input from combined numerical + embedded categorical features
                input_dim = first_layer_input_dim
                output_dim = layer_sizes[i]
            else:
                # Subsequent layers: input from previous layer
                input_dim = layer_sizes[i - 1]
                output_dim = layer_sizes[i]

            # Create linear layer
            layer = nnx.Linear(
                in_features=input_dim, out_features=output_dim, rngs=rngs
            )
            self.layers.append(layer)

            if i < self.num_layers - 1:
                # Add batch normalization layer for all but the last layer
                bn_layer = nnx.BatchNorm(
                    num_features=output_dim,
                    # `use_running_average` will be set dynamically in the forward pass
                    rngs=rngs,
                )
                self.bns.append(bn_layer)

        # Create dropout layer for regularization
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, training: bool = False):
        """
        Forward pass through the MLP.

        Args:
            x: Input array of shape (batch_size, num_features)
               Categorical features are in the rightmost num_categorical columns
            training: Whether model is in training mode (affects dropout)

        Returns:
            Regression predictions of shape (batch_size, 1)
        """
        # batch_size = x.shape[0]

        # Split input into numerical and categorical features
        if self.num_categorical > 0:
            # Numerical features are the leftmost columns
            numerical_features = x[
                :, : self.num_numerical
            ]  # Shape: (batch_size, num_numerical)

            # Categorical features are the rightmost columns
            categorical_features = x[
                :, self.num_numerical:
            ]  # Shape: (batch_size, num_categorical)

            # Apply embeddings to each categorical feature
            embedded_categoricals = []
            for i in range(self.num_categorical):
                # Extract the i-th categorical feature column
                cat_feature = categorical_features[:, i].astype(
                    jnp.int32
                )  # Shape: (batch_size,)

                # Apply embedding layer to get embedding vectors
                embedded = self.embeddings[i](
                    cat_feature
                )  # Shape: (batch_size, embed_dim)
                embedded_categoricals.append(embedded)

            # Concatenate all categorical embeddings
            if embedded_categoricals:
                cat_embeddings = jnp.concatenate(
                    embedded_categoricals, axis=1
                )  # Shape: (batch_size, num_categorical * embed_dim)

                # Combine numerical features with categorical embeddings
                combined_features = jnp.concatenate(
                    [numerical_features, cat_embeddings], axis=1
                )
            else:
                combined_features = numerical_features
        else:
            # No categorical features, use all features as numerical
            combined_features = x

        # Pass through MLP layers
        output = combined_features

        for i, layer in enumerate(self.layers):
            # Apply linear transformation
            output = layer(output)

            # Apply ReLU activation to all layers except the last one
            # Last layer is linear for regression output
            if i < len(self.layers) - 1:

                # NEW: Apply BatchNorm after the linear transformation and before activation.
                # `use_running_average` is set to False during training and True during inference.
                output = self.bns[i](output, use_running_average=not training)

                output = nnx.relu(output)

                # Apply dropout during training for regularization
                if training:
                    output = self.dropout(output, deterministic=False)
                else:
                    output = self.dropout(output, deterministic=True)

        # Final output for regression (no activation on last layer)
        return output


class EnsembleModel(nnx.Module):
    def __init__(self, models: list):
        super().__init__()

        self.models = models

    def __call__(self, x, training: bool = False, num_models: int = None):  
        # Get predictions from each model
        
        if num_models >= len(self.models):
            raise ValueError("num_models must be less than the number of models in the ensemble.")
        
        predictions = []
        for i in range(num_models):
            model = self.models[i]
            # Ensure each model is called with the same input and training flag
            predictions.append(model(x, training=training))
            
        
        # Average the predictions
        ensemble_prediction = jnp.mean(jnp.array(predictions), axis=0)
        
        # return the averaged predictions
        return ensemble_prediction


class MLPRegressor_v3(nnx.Module):
    """
    Multi-Layer Perceptron for regression tasks with flexible categorical feature embeddings.

    This model handles both numerical and categorical features by:
    1. Applying embeddings to categorical features with customizable dimensions per feature
    2. Concatenating numerical features with categorical embeddings
    3. Passing the combined features through MLP layers with batch normalization
    4. Outputting a single regression value
    """

    def __init__(
        self,
        num_features: int,
        layer_sizes: Tuple[int, ...],
        categorical_configs: list,  # List of (vocab_size, embed_dim) tuples
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the MLP Regressor with flexible categorical embeddings.
    
        Args:
            num_features: Total number of input features (numerical + categorical)
            layer_sizes: Tuple defining the layer dimensions
            categorical_configs: List of (vocab_size, embed_dim) tuples for each categorical feature
                                Format: [(vocab_size_1, embed_dim_1), (vocab_size_2, embed_dim_2), ...]
                                If empty, all features are treated as numerical
            dropout_rate: Dropout rate for regularization
            rngs: Random number generator state
        """
        # Store configuration parameters
        self.num_features = num_features
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.categorical_configs = categorical_configs
        self.num_categorical = len(categorical_configs)
        self.dropout_rate = dropout_rate
    
        # Calculate number of numerical features
        # Categorical features are at the rightmost columns
        self.num_numerical = num_features - self.num_categorical
    
        # Create embedding layers with explicit naming to preserve order
        self.embeddings = {}
        total_embed_dim = 0
    
        for i, (vocab_size, embed_dim) in enumerate(categorical_configs):
            embed_layer = nnx.Embed(
                num_embeddings=vocab_size,
                features=embed_dim,
                rngs=rngs,
            )
            # Use explicit string keys to ensure consistent ordering
            self.embeddings[f"categorical_{i}"] = embed_layer
            total_embed_dim += embed_dim
    
        # Calculate input dimension for first MLP layer
        # = numerical features + sum of all categorical embedding dimensions
        first_layer_input_dim = self.num_numerical + total_embed_dim
    
        # Create MLP layers based on layer_sizes specification
        self.layers = []
    
        # List of batch normalization layers
        self.bns = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = first_layer_input_dim
                output_dim = layer_sizes[i]
            else:
                input_dim = layer_sizes[i - 1]
                output_dim = layer_sizes[i]
    
            # Create linear layer
            layer = nnx.Linear(
                in_features=input_dim, out_features=output_dim, rngs=rngs
            )
            self.layers.append(layer)
    
            if i < self.num_layers - 1:
                bn_layer = nnx.BatchNorm(num_features=output_dim, rngs=rngs)
                self.bns.append(bn_layer)
    
        # Create dropout layer for regularization
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    
    def __call__(self, x, training: bool = False):
        """
        Forward pass through the MLP.
    
        Args:
            x: Input array of shape (batch_size, num_features)
               Categorical features are in the rightmost num_categorical columns
            training: Whether model is in training mode (affects dropout and batch norm)
    
        Returns:
            Regression predictions of shape (batch_size, 1)
        """
        import jax.numpy as jnp
        
        # Split input into numerical and categorical features
        if self.num_categorical > 0:
            # Numerical features are the leftmost columns
            numerical_features = x[
                :, : self.num_numerical
            ]  # Shape: (batch_size, num_numerical)
    
            # Categorical features are the rightmost columns
            categorical_features = x[
                :, self.num_numerical:
            ]  # Shape: (batch_size, num_categorical)
    
            # Apply embeddings to each categorical feature with explicit ordering
            embedded_categoricals = []
            for i in range(self.num_categorical):
                cat_feature = categorical_features[:, i].astype(jnp.int32)
                # Use consistent key naming to ensure proper ordering
                embedded = self.embeddings[f"categorical_{i}"](cat_feature)
                embedded_categoricals.append(embedded)
    
            # Concatenate all categorical embeddings
            if embedded_categoricals:
                cat_embeddings = jnp.concatenate(
                    embedded_categoricals, axis=1
                )  # Shape: (batch_size, total_embed_dim)
    
                # Combine numerical features with categorical embeddings
                combined_features = jnp.concatenate(
                    [numerical_features, cat_embeddings], axis=1
                )
            else:
                combined_features = numerical_features
        else:
            # No categorical features, use all features as numerical
            combined_features = x
    
        # Pass through MLP layers with batch normalization
        output = combined_features
    
        for i, layer in enumerate(self.layers):
            # Apply linear transformation
            output = layer(output)
    
            # Apply batch norm, activation, and dropout to all layers except the last one
            # Last layer is linear for regression output
            if i < len(self.layers) - 1:
                # Apply batch normalization
                output = self.bns[i](output, use_running_average=not training)
                # Apply ReLU activation
                output = nnx.relu(output)
                # Apply dropout
                output = self.dropout(output, deterministic=not training)
    
        # Final output for regression (no activation on last layer)
        return output
    

class MLP_v1(nnx.Module):
    """
    Multi-Layer Perceptron for regression tasks with flexible categorical feature embeddings.

    This model handles both numerical and categorical features by:
    1. Applying embeddings to categorical features with customizable dimensions per feature
    2. Concatenating numerical features with categorical embeddings
    3. Passing the combined features through MLP layers with batch normalization
    """

    def __init__(
        self,
        num_features: int,
        layer_sizes: Tuple[int, ...],
        categorical_configs: list,  # List of (vocab_size, embed_dim) tuples
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the MLP Regressor with flexible categorical embeddings.
    
        Args:
            num_features: Total number of input features (numerical + categorical)
            layer_sizes: Tuple defining the layer dimensions
            categorical_configs: List of (vocab_size, embed_dim) tuples for each categorical feature
                                Format: [(vocab_size_1, embed_dim_1), (vocab_size_2, embed_dim_2), ...]
                                If empty, all features are treated as numerical
            dropout_rate: Dropout rate for regularization
            rngs: Random number generator state
        """
        # Store configuration parameters
        self.num_features = num_features
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.categorical_configs = categorical_configs
        self.num_categorical = len(categorical_configs)
        self.dropout_rate = dropout_rate
    
        # Calculate number of numerical features
        # Categorical features are at the rightmost columns
        self.num_numerical = num_features - self.num_categorical
    
        # Create embedding layers with explicit naming to preserve order
        self.embeddings = {}
        total_embed_dim = 0
    
        for i, (vocab_size, embed_dim) in enumerate(categorical_configs):
            embed_layer = nnx.Embed(
                num_embeddings=vocab_size,
                features=embed_dim,
                rngs=rngs,
            )
            # Use explicit string keys to ensure consistent ordering
            self.embeddings[f"categorical_{i}"] = embed_layer
            total_embed_dim += embed_dim
    
        # Calculate input dimension for first MLP layer
        # = numerical features + sum of all categorical embedding dimensions
        first_layer_input_dim = self.num_numerical + total_embed_dim
    
        # Create MLP layers based on layer_sizes specification with explicit naming
        self.layers = {}
    
        # List of batch normalization layers
        self.bns = {}
        for i in range(self.num_layers):
            if i == 0:
                input_dim = first_layer_input_dim
                output_dim = layer_sizes[i]
            else:
                input_dim = layer_sizes[i - 1]
                output_dim = layer_sizes[i]
    
            # Create linear layer
            layer = nnx.Linear(
                in_features=input_dim, out_features=output_dim, rngs=rngs
            )
            self.layers[f"layer_{i}"] = layer
    
            if i < self.num_layers - 1:
                bn_layer = nnx.BatchNorm(num_features=output_dim, rngs=rngs)
                self.bns[f"bn_{i}"] = bn_layer
    
        # Create dropout layer for regularization
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    
    def __call__(self, x, training: bool = False):
        """
        Forward pass through the MLP.
    
        Args:
            x: Input array of shape (batch_size, num_features)
               Categorical features are in the rightmost num_categorical columns
            training: Whether model is in training mode (affects dropout and batch norm)
    
        Returns:
            Regression predictions of shape (batch_size, 1)
        """
        
        # Split input into numerical and categorical features
        if self.num_categorical > 0:
            # Numerical features are the leftmost columns
            numerical_features = x[
                :, : self.num_numerical
            ]  # Shape: (batch_size, num_numerical)
    
            # Categorical features are the rightmost columns
            categorical_features = x[
                :, self.num_numerical:
            ]  # Shape: (batch_size, num_categorical)
    
            # Apply embeddings to each categorical feature with explicit ordering
            embedded_categoricals = []
            for i in range(self.num_categorical):
                cat_feature = categorical_features[:, i].astype(jnp.int32)
                # Use consistent key naming to ensure proper ordering
                embedded = self.embeddings[f"categorical_{i}"](cat_feature)
                embedded_categoricals.append(embedded)
    
            # Concatenate all categorical embeddings
            if embedded_categoricals:
                cat_embeddings = jnp.concatenate(
                    embedded_categoricals, axis=1
                )  # Shape: (batch_size, total_embed_dim)
    
                # Combine numerical features with categorical embeddings
                combined_features = jnp.concatenate(
                    [numerical_features, cat_embeddings], axis=1
                )
            else:
                combined_features = numerical_features
        else:
            # No categorical features, use all features as numerical
            combined_features = x
    
        # Pass through MLP layers with batch normalization
        output = combined_features

        for i in range(self.num_layers):
            layer = self.layers[f"layer_{i}"]
            
            # Apply linear transformation
            output = layer(output)
            # Apply batch norm, activation and dropout to all layers except the last one
            
            if i < self.num_layers - 1:
                # Apply batch normalization
                output = self.bns[f"bn_{i}"](output, use_running_average=not training)

                # Apply ReLU activation
                output = nnx.relu(output)

                # Apply dropout during training for regularization
                output = self.dropout(output, deterministic=not training)

        return output
    

class ResidualBlock2Layer(nnx.Module):
    """
    A residual block for use in ResNet architectures.
    
    This block consists of two linear layers with ReLU activations and a skip connection.
    It can be used to build deeper networks while mitigating vanishing gradient issues.
    """

    def __init__(self, in_features: int, out_features: int, dropout_rate: float, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(in_features, out_features, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.linear2 = nnx.Linear(out_features, out_features, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        
        # Projection for the residual if dimensions don't match
        if in_features != out_features:
            self.shortcut = nnx.Linear(in_features, out_features, rngs=rngs)
        else:
            self.shortcut = None

    def __call__(self, x, training: bool = False):
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        residual = x  # Save the input for the skip connection
        
        x = self.linear1(x)  # First linear layer
        x = self.bn1(x, use_running_average=not training)  # Batch normalization
        x = nnx.relu(x)  # ReLU activation
        x = self.dropout(x, deterministic=not training)  # Apply dropout
        
        x = self.linear2(x)  # Second linear layer
        x = self.bn2(x, use_running_average=not training)  # Batch normalization
        
       # Skip connection
        if self.shortcut:
            residual = self.shortcut(residual)

        x = x + residual  # Add skip connection
        x = nnx.relu(x)  # Apply ReLU activation after adding skip connection

        
        return x  # Output tensor of shape (batch_size, out_features)


class ResidualBlock3Layer(nnx.Module):
    """
    A residual block for use in ResNet architectures with three linear layers.
    
    This block consists of three linear layers with ReLU activations and a skip connection.
    It can be used to build deeper networks while mitigating vanishing gradient issues.
    """

    def __init__(self, in_features: int, mid_features: int, out_features: int, dropout_rate: float, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(in_features, mid_features, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=mid_features, rngs=rngs)
        self.linear2 = nnx.Linear(mid_features, mid_features, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=mid_features, rngs=rngs)
        self.linear3 = nnx.Linear(mid_features, out_features, rngs=rngs)
        self.bn3 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        
        # Projection for the residual if dimensions don't match
        if in_features != out_features:
            self.shortcut = nnx.Linear(in_features, out_features, rngs=rngs)
        else:
            self.shortcut = None

    def __call__(self, x, training: bool = False):
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        residual = x  # Save the input for the skip connection
        
        x = self.linear1(x)  # First linear layer
        x = self.bn1(x, use_running_average=not training)  # Batch normalization
        x = nnx.relu(x)  # ReLU activation
        x = self.dropout(x, deterministic=not training)  # Apply dropout
        
        x = self.linear2(x)  # Second linear layer
        x = self.bn2(x, use_running_average=not training)  # Batch normalization
        x = nnx.relu(x)  # ReLU activation
        x = self.dropout(x, deterministic=not training)  # Apply dropout
        
        x = self.linear3(x)  # Third linear layer
        x = self.bn3(x, use_running_average=not training)  # Batch normalization
        
       # Skip connection
        if self.shortcut:
            residual = self.shortcut(residual)
            
        x = x + residual  # Add skip connection
        x = nnx.relu(x)  # Apply ReLU activation after adding skip connection
        
        return x  # Output tensor of shape (batch_size, out_features)
    
    
class ResNet_v1(nnx.Module):
    """
    A simple ResNet architecture using residual blocks.
    
    This model consists of multiple residual blocks followed by a final linear layer for regression.
    It can be used for tasks like time series forecasting or other regression problems.
    """

    def __init__(
        self,
        num_features: int,
        resblock_configs: list[Tuple[int, ...]],  # List of (block_type, out_features) tuples for each residual block block_type can be 2 or 3 for ResidualBlock2Layer or ResidualBlock3Layer
        categorical_configs: list,  # List of (vocab_size, embed_dim) tuples
        output_dim: int,  # Output dimension for the final regression layer
        dropout_rate: float,  # Dropout rate for regularization
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.num_features = num_features
        self.resblock_configs = resblock_configs
        self.categorical_configs = categorical_configs
        self.output_dim = output_dim
        self.num_categorical = len(categorical_configs)
        self.num_blocks = len(resblock_configs)
        
        # Calculate number of numerical features
        # Categorical features are at the rightmost columns
        self.num_numerical = num_features - self.num_categorical
    
        # Create embedding layers with explicit naming to preserve order
        self.embeddings = {}
        total_embed_dim = 0
    
        for i, (vocab_size, embed_dim) in enumerate(categorical_configs):
            embed_layer = nnx.Embed(
                num_embeddings=vocab_size,
                features=embed_dim,
                rngs=rngs,
            )
            # Use explicit string keys to ensure consistent ordering
            self.embeddings[f"categorical_{i}"] = embed_layer
            total_embed_dim += embed_dim
    
        # Calculate input dimension for first MLP layer
        # = numerical features + sum of all categorical embedding dimensions
        first_layer_input_dim = self.num_numerical + total_embed_dim        
        
        # Create residual blocks based on the configuration
        self.resblocks = {}
        
        for i, (block_type, out_features) in enumerate(resblock_configs):
            if block_type == 2:
                # ResidualBlock2Layer
                block = ResidualBlock2Layer(
                    in_features=first_layer_input_dim if i == 0 else resblock_configs[i-1][1],
                    out_features=out_features,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
            elif block_type == 3:
                # ResidualBlock3Layer
                mid_features = out_features // 2
                block = ResidualBlock3Layer(
                    in_features=first_layer_input_dim if i == 0 else resblock_configs[i-1][1],
                    mid_features=mid_features,
                    out_features=out_features,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
            else:
                raise ValueError(f"Unsupported block type: {block_type}. Must be 2 or 3.")
            
            self.resblocks[f"block_{i}"] = block
            
        # Final linear layer for regression/classification
        self.final_linear = nnx.Linear(
            in_features=resblock_configs[-1][1],  # Input from the last residual block
            out_features=self.output_dim,
            rngs=rngs,
        )
        


    def __call__(self, x, training: bool = False):
        """
        Forward pass through the ResNet model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            training: Whether model is in training mode (affects batch normalization)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        
        # Split input into numerical and categorical features
        if self.num_categorical > 0:
            # Numerical features are the leftmost columns
            numerical_features = x[
                :, : self.num_numerical
            ]  # Shape: (batch_size, num_numerical)
    
            # Categorical features are the rightmost columns
            categorical_features = x[
                :, self.num_numerical:
            ]  # Shape: (batch_size, num_categorical)
    
            # Apply embeddings to each categorical feature with explicit ordering
            embedded_categoricals = []
            for i in range(self.num_categorical):
                cat_feature = categorical_features[:, i].astype(jnp.int32)
                # Use consistent key naming to ensure proper ordering
                embedded = self.embeddings[f"categorical_{i}"](cat_feature)
                embedded_categoricals.append(embedded)
    
            # Concatenate all categorical embeddings
            if embedded_categoricals:
                cat_embeddings = jnp.concatenate(
                    embedded_categoricals, axis=1
                )  # Shape: (batch_size, total_embed_dim)
    
                # Combine numerical features with categorical embeddings
                combined_features = jnp.concatenate(
                    [numerical_features, cat_embeddings], axis=1
                )
            else:
                combined_features = numerical_features
        else:
            # No categorical features, use all features as numerical
            combined_features = x
    
        # Pass through MLP layers with batch normalization
        output = combined_features

        for i in range(self.num_blocks):
            res_block = self.resblocks[f"block_{i}"]
            
            # Apply the residual block
            output = res_block(output, training=training)

        # Final linear layer for regression/classification
        output = self.final_linear(output)  # Shape: (batch_size, output_dim)

        return output
