"""
    This module contains custom Flax transformers for machine learning tasks.
"""

import jax.numpy as jnp
from flax import nnx
import jax

class KVCache(nnx.Module):
    """A container for Key-Value cache state for a single attention module."""
    def __init__(self):
        # We use nnx.Variable for mutable state that is not a trainable parameter.
        # It's initialized to None and will be populated with a JAX array later.
        self.key = nnx.Variable(None)
        self.value = nnx.Variable(None)
        
        
class FFN(nnx.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN) sub-layer.

    This module consists of two linear transformations with a ReLU activation
    in between. It is applied to each position separately and identically.
    The primary role of the FFN is to provide non-linear transformations and
    increase the model's representational capacity.

    As per the "Attention is All You Need" paper, the FFN expands the input
    dimension `d_model` to a larger inner dimension `d_ff` and then contracts
    it back to `d_model`.
    """
    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        """
        Initializes the Feed-Forward Network module.

        Args:
            d_model: The dimensionality of the input and output.
            d_ff: The dimensionality of the inner layer. Typically 4 * d_model.
            rngs: The JAX random number generators required by Flax NNX.
        """
        # The first linear layer expands the input from d_model to d_ff.
        self.linear1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        
        # The second linear layer contracts the representation back to d_model.
        self.linear2 = nnx.Linear(d_ff, d_model, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Performs the forward pass for the FFN.

        Note: The `training` argument is included for API consistency with other
        layers, but is not used here as dropout is handled by the parent layer.

        Args:
            x: The input tensor. Shape: (batch_size, seq_len, d_model).
            training: A boolean indicating if the model is in training mode.

        Returns:
            The output tensor. Shape: (batch_size, seq_len, d_model).
        """
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        # Dropout is handled by the parent TransformerEncoderLayer/DecoderLayer.
        return x


class MultiHeadAttention(nnx.Module):
    """
    Implements the Multi-Head Attention mechanism.

    This module computes scaled dot-product attention in parallel across several
    "heads". It is designed to be a generic sub-layer that can be used for
    self-attention (where query, key, and value are the same) and for
    cross-attention (where query comes from a decoder, and key/value come
    from an encoder).

    As per the "Attention is All You Need" paper, this module handles the
    dropout for the attention weights internally. It expects the parent layer
    (e.g., TransformerEncoderLayer) to handle the residual connection and the
    dropout on its final output.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
        """
        Initializes the Multi-Head Attention module.

        Args:
            d_model: The total dimensionality of the model's embeddings.
            num_heads: The number of parallel attention heads. Must divide d_model.
            dropout_rate: The dropout rate to apply to the attention weights after softmax.
            rngs: The JAX random number generators required by Flax NNX.
        """
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # Dimension of each head's Q, K, V

        # --- Learnable Layers ---

        # A single, large linear layer is used for each of Q, K, and V for computational
        # efficiency, rather than creating separate layers for each head.
        self.query_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.key_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.value_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        # The final linear layer that combines the outputs of all heads.
        self.output_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        
        # --- Internal Dropout for Attention Weights ---
        # This is applied to the attention scores after the softmax operation.
        self.attn_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        
        # --- KV Cache for Efficient Inference ---
        self.cache = KVCache()

    def init_cache(self, batch_size: int, max_seq_len: int):
        """
        Initializes the Key-Value cache with zero-filled tensors.

        This method should be called once at the beginning of an inference session
        to pre-allocate memory for the cache.
        
        Args:
            batch_size: The batch size for the inference session.
            max_seq_len: The maximum sequence length the cache should support.
        """
        # The cache shape is (batch, num_heads, max_seq_len, d_head)
        key_shape = (batch_size, self.num_heads, max_seq_len, self.d_head)
        value_shape = (batch_size, self.num_heads, max_seq_len, self.d_head)
        
        # Initialize the mutable Variables within the cache.
        self.cache.key.value = jnp.zeros(key_shape)
        self.cache.value.value = jnp.zeros(value_shape)
        
    def generate_step(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, decode_step_index: int):
        """
        Performs a single, cached, autoregressive step for inference.

        This method is designed for self-attention within the decoder, where it
        updates and uses a dynamic cache.

        Args:
            query: The query tensor for the current single token. Shape: (batch, 1, d_model).
            key: The key tensor for the current single token. Shape: (batch, 1, d_model).
            value: The value tensor for the current single token. Shape: (batch, 1, d_model).
            decode_step_index: The current time-step in the generation loop, used
                               to index the cache.

        Returns:
            The output of the attention mechanism for the single input token.
            Shape: (batch_size, 1, d_model).
        """
        batch_size = query.shape[0]

        # 1. Project the new Q, K, V for the single input token.
        q_new = self.query_proj(query)
        k_new = self.key_proj(key)
        v_new = self.value_proj(value)

        # 2. Reshape the new Q, K, V to a per-head representation.
        # Shape becomes: (batch, num_heads, 1, d_head)
        q_heads = q_new.reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        k_heads_new = k_new.reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        v_heads_new = v_new.reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # 3. Update the cache at the current index with the new K and V.
        # The .at[...].set(...) pattern is JAX's syntax for an out-of-place update.
        self.cache.key.value = self.cache.key.value.at[:, :, decode_step_index, :].set(k_heads_new.squeeze(2))
        self.cache.value.value = self.cache.value.value.at[:, :, decode_step_index, :].set(v_heads_new.squeeze(2))

        # 4. Retrieve the full history of Keys and Values from the cache.
        # We slice up to and including the current step index.
        k_heads_all = self.cache.key.value[:, :, :decode_step_index + 1, :]
        v_heads_all = self.cache.value.value[:, :, :decode_step_index + 1, :]

        # 5. Perform scaled dot-product attention using the new Query and full K, V history.
        scaling_factor = jnp.sqrt(self.d_head)
        scores = (q_heads @ k_heads_all.transpose(0, 1, 3, 2)) / scaling_factor
        
        # NOTE: Causal masking is implicitly handled by only attending to the history
        # in the cache up to the current step. No explicit mask is needed here.

        attn_weights = jax.nn.softmax(scores, axis=-1)
        # Dropout is disabled during inference.

        # 6. Apply attention to the full Value history.
        output = attn_weights @ v_heads_all
        
        # 7. Reshape and project the single output token.
        output_reshaped = output.transpose(0, 2, 1, 3).reshape(batch_size, 1, self.d_model)
        final_output = self.output_proj(output_reshaped)
        
        return final_output

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, mask: jnp.ndarray | None = None, training: bool = False):
        """
        Performs the forward pass for multi-head attention.

        Args:
            query: The query tensor. Shape: (batch_size, q_seq_len, d_model).
            key: The key tensor. Shape: (batch_size, kv_seq_len, d_model).
            value: The value tensor. Shape: (batch_size, kv_seq_len, d_model).
            mask: An optional mask to prevent attention to certain positions (e.g.,
                  padding tokens or future tokens in causal attention).
                  Shape should broadcast to (batch_size, num_heads, q_seq_len, kv_seq_len).
            training: A boolean indicating if the model is in training mode, which
                      determines whether dropout is active.

        Returns:
            The output of the attention mechanism. Shape: (batch_size, q_seq_len, d_model).
        """
        batch_size = query.shape[0]

        # 1. Project Q, K, V into their respective spaces.
        # The output of each is (batch_size, seq_len, d_model).
        q_proj = self.query_proj(query)
        k_proj = self.key_proj(key)
        v_proj = self.value_proj(value)

        # 2. Reshape and transpose for multi-head processing.
        # This splits the d_model dimension into num_heads and d_head.
        # Shape becomes: (batch_size, num_heads, seq_len, d_head).
        q_heads = q_proj.reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        k_heads = k_proj.reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        v_heads = v_proj.reshape(batch_size, -1, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # 3. Perform scaled dot-product attention.
        # (B, H, qS, dH) @ (B, H, dH, kvS) -> (B, H, qS, kvS)
        scaling_factor = jnp.sqrt(self.d_head)
        scores = (q_heads @ k_heads.transpose(0, 1, 3, 2)) / scaling_factor
        
        # Apply the mask if it is provided. The mask is typically 0 for positions
        # to be masked and 1 otherwise. We set masked positions to a very large
        # negative number so they become zero after softmax.
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Apply softmax along the last axis to get attention probabilities.
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply dropout to the attention weights.
        attn_weights = self.attn_dropout(attn_weights, deterministic=not training)

        # 4. Apply the computed attention weights to the value vectors.
        # (B, H, qS, kvS) @ (B, H, kvS, dH) -> (B, H, qS, dH)
        output = attn_weights @ v_heads

        # 5. Concatenate heads and apply the final linear projection.
        # First, transpose and reshape to bring the heads together.
        # (B, H, qS, dH) -> (B, qS, H, dH) -> (B, qS, D)
        output_reshaped = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # Final linear projection.
        final_output = self.output_proj(output_reshaped)

        return final_output
        

class TransformerEncoder(nnx.Module):
    """
    A stack of N identical Transformer Encoder Layers.

    This module processes the sequence of input embeddings by passing them
    through a series of `TransformerEncoderLayer` modules in a guaranteed
    sequential order. It's responsible for creating the final contextual
    representation of the input sequence (the "encoder context" or "memory"),
    which is then used by the decoder.

    The layers are stored in a dictionary and accessed via an explicit loop
    to ensure deterministic execution and robust serialization.
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float, *, rngs: nnx.Rngs):
        """
        Initializes the complete Transformer Encoder.

        Args:
            num_layers: The number of `TransformerEncoderLayer` modules to stack.
            d_model: The dimensionality of the model's embeddings.
            num_heads: The number of heads for the multi-head attention modules.
            d_ff: The inner dimension of the feed-forward networks.
            dropout_rate: The dropout rate used in the encoder layers.
            rngs: The JAX random number generators required by Flax NNX.
        """
        # Store the number of layers as an explicit attribute for deterministic iteration.
        self.num_layers = num_layers
        
        # Create a dictionary to hold the N identical encoder layers,
        # providing explicit names for each layer.
        self.layers = {
            f"layer_{i}": TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                rngs=rngs
            )
            for i in range(self.num_layers)
        }
        
        # A final layer normalization is applied after the entire stack.
        self.norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None, training: bool = False):
        """
        Performs the forward pass for the entire encoder stack.

        Args:
            x: The input tensor, typically token embeddings + positional encodings.
               Shape: (batch_size, seq_len, d_model).
            mask: The padding mask for the input sequence.
            training: A boolean indicating if the model is in training mode.

        Returns:
            The encoder's final output tensor. Shape: (batch_size, seq_len, d_model).
        """
        # Sequentially pass the input through each layer in the stack.
        # This loop guarantees the layers are processed in the correct order (0, 1, 2, ...).
        for i in range(self.num_layers):
            layer = self.layers[f"layer_{i}"]
            x = layer(x, mask=mask, training=training)
        
        # Apply the final layer normalization.
        # This provides a clean, stabilized output to the decoder.
        return self.norm(x)


class TransformerEncoderLayer(nnx.Module):
    """
    A single layer of the Transformer Encoder, containing a Multi-Head Attention
    sub-layer and a Feed-Forward Network sub-layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate, rngs=rngs)
        self.ffn = FFN(
            d_model=d_model, 
            d_ff=d_ff,
            rngs=rngs
            )
        
        # We need two LayerNorms and two Dropouts for the two sub-layers
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None, training: bool):
        # 1. First Sub-layer: Multi-Head Self-Attention
        attn_output = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            mask=mask, 
            training=training
        )
        # Apply dropout, then the residual connection, then layer normalization
        # This exactly follows the formula: LayerNorm(x + Dropout(Sublayer(x)))
        x = self.norm1(x + self.dropout1(attn_output, deterministic=not training))
        
        # 2. Second Sub-layer: Feed-Forward Network
        ffn_output = self.ffn(x, training=training)
        # Apply dropout, residual connection, and layer normalization again
        x = self.norm2(x + self.dropout2(ffn_output, deterministic=not training))
        
        return x
    

class TransformerDecoderLayer(nnx.Module):
    """
    A single layer of the Transformer Decoder.

    This module encapsulates one complete processing block of the decoder, which
    consists of three main sub-layers:
    1. A Masked (Causal) Multi-Head Self-Attention mechanism.
    2. A Multi-Head Cross-Attention mechanism that attends to the encoder's output.
    3. A Position-wise Feed-Forward Network (FFN).

    Each sub-layer is followed by a residual connection and layer normalization,
    as described by the formula: `LayerNorm(x + Dropout(Sublayer(x)))`.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float, *, rngs: nnx.Rngs):
        """
        Initializes the Transformer Decoder Layer.

        Args:
            d_model: The dimensionality of the model's embeddings.
            num_heads: The number of heads for both attention modules.
            d_ff: The inner dimension of the feed-forward network.
            dropout_rate: The dropout rate to be applied after each sub-layer.
            rngs: The JAX random number generators required by Flax NNX.
        """
        # --- Sub-layer Modules ---
        # The first sub-layer is for causal self-attention on the target sequence.
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate, rngs=rngs)
        
        # The second sub-layer is for cross-attention, attending to the encoder context.
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout_rate, rngs=rngs)
        
        # The third sub-layer is the position-wise feed-forward network.
        self.ffn = FFN(d_model, d_ff, rngs=rngs)
        
        # --- Layer Normalization and Dropout ---
        # We need three of each for the three sub-layers.
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        
        # --- Cache for Cross-Attention ---
        # We store the pre-computed Key and Value from the encoder context here
        # to avoid recalculating them at every decoding step.
        self.cached_cross_key = nnx.Variable(None)
        self.cached_cross_value = nnx.Variable(None)

    def init_cache(self, batch_size: int, max_seq_len: int, encoder_context: jnp.ndarray):
        """
        Initializes all caches required for efficient inference.

        This method pre-computes the static cross-attention keys and values
        and pre-allocates space for the dynamic self-attention cache.

        Args:
            batch_size: The batch size for the inference session.
            max_seq_len: The maximum sequence length for the generation.
            encoder_context: The final output of the encoder.
        """
        # 1. Initialize the dynamic cache for the self-attention module.
        self.self_attn.init_cache(batch_size, max_seq_len)

        # 2. Pre-compute and cache the static Key and Value for cross-attention.
        # This is a one-time computation that saves significant time during generation.
        self.cached_cross_key.value = self.cross_attn.key_proj(encoder_context)
        self.cached_cross_value.value = self.cross_attn.value_proj(encoder_context)

    def generate_step(self, 
                      y: jnp.ndarray, 
                      cross_attn_mask: jnp.ndarray | None, 
                      decode_step_index: int):
        """
        Performs a single, cached, autoregressive step for inference.

        Args:
            y: The input tensor for the single current token. Shape: (batch, 1, d_model).
            cross_attn_mask: The padding mask for the source sequence.
            decode_step_index: The current time-step in the generation loop.

        Returns:
            The output tensor for the single token. Shape: (batch, 1, d_model).
        """
        # 1. Self-Attention with dynamic cache
        self_attn_output = self.self_attn.generate_step(
            query=y, 
            key=y, 
            value=y,
            decode_step_index=decode_step_index
        )
        y = self.norm1(y + self.dropout1(self_attn_output, deterministic=True))

        # 2. Cross-Attention with static cache
        # We re-use the standard __call__ method here, but pass in the pre-computed
        # key and value matrices that we cached in the init_cache step.
        cross_attn_output = self.cross_attn(
            query=y, 
            key=self.cached_cross_key.value, 
            value=self.cached_cross_value.value,
            mask=cross_attn_mask,
            training=False
        )
        y = self.norm2(y + self.dropout2(cross_attn_output, deterministic=True))
        
        # 3. Feed-Forward Network
        ffn_output = self.ffn(y, training=False)
        y = self.norm3(y + self.dropout3(ffn_output, deterministic=True))

        return y

    def __call__(self, 
                 y: jnp.ndarray, 
                 encoder_context: jnp.ndarray, 
                 self_attn_mask: jnp.ndarray | None, 
                 cross_attn_mask: jnp.ndarray | None, 
                 training: bool):
        """
        Performs the forward pass for one decoder layer.

        Args:
            y: The input tensor from the previous decoder layer.
               Shape: (batch_size, target_seq_len, d_model).
            encoder_context: The final output from the encoder stack.
                             Shape: (batch_size, source_seq_len, d_model).
            self_attn_mask: The combined causal and padding mask for self-attention.
            cross_attn_mask: The padding mask for the source sequence for cross-attention.
            training: A boolean indicating if the model is in training mode.

        Returns:
            The output tensor of the same shape as the input `y`.
        """
        # --- First Sub-layer: Causal Multi-Head Self-Attention ---
        # The decoder attends to itself, but is prevented from seeing future tokens.
        self_attn_output = self.self_attn(
            query=y, 
            key=y, 
            value=y, 
            mask=self_attn_mask, 
            training=training
        )
        # Residual connection with dropout and layer normalization.
        y = self.norm1(y + self.dropout1(self_attn_output, deterministic=not training))
        
        # --- Second Sub-layer: Multi-Head Cross-Attention ---
        # The decoder attends to the encoder's output (the context).
        # Query comes from the decoder's state; Key and Value come from the encoder.
        cross_attn_output = self.cross_attn(
            query=y, 
            key=encoder_context, 
            value=encoder_context, 
            mask=cross_attn_mask, 
            training=training
        )
        # Residual connection with dropout and layer normalization.
        y = self.norm2(y + self.dropout2(cross_attn_output, deterministic=not training))

        # --- Third Sub-layer: Feed-Forward Network ---
        ffn_output = self.ffn(y, training=training)
        # Residual connection with dropout and layer normalization again.
        y = self.norm3(y + self.dropout3(ffn_output, deterministic=not training))
        
        return y
    
    
# In transformer_components.py

class TransformerDecoder(nnx.Module):
    """
    A stack of N identical Transformer Decoder Layers.

    This module processes the sequence of target embeddings by passing them
    through a series of `TransformerDecoderLayer` modules. In each layer,
    it performs both self-attention on the target sequence and cross-attention
    on the final output from the encoder (`encoder_context`).

    The layers are stored in a dictionary and accessed via an explicit loop
    to ensure deterministic execution and robust serialization.
    """
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float, *, rngs: nnx.Rngs):
        """
        Initializes the complete Transformer Decoder.

        Args:
            num_layers: The number of `TransformerDecoderLayer` modules to stack.
            d_model: The dimensionality of the model's embeddings.
            num_heads: The number of heads for the attention modules.
            d_ff: The inner dimension of the feed-forward networks.
            dropout_rate: The dropout rate used in the decoder layers.
            rngs: The JAX random number generators required by Flax NNX.
        """
        # Store the number of layers as an explicit attribute for deterministic iteration.
        self.num_layers = num_layers
        
        # Create a dictionary to hold the N identical decoder layers,
        # providing explicit names for each layer.
        self.layers = {
            f"layer_{i}": TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                rngs=rngs
            )
            for i in range(self.num_layers)
        }
        
        # A final layer normalization is applied after the entire stack.
        self.norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def init_cache(self, batch_size: int, max_seq_len: int, encoder_context: jnp.ndarray):
        """
        Initializes the caches for all layers in this decoder.

        This method delegates the cache initialization to each individual
        `TransformerDecoderLayer`, which will pre-compute its cross-attention
        K/V and pre-allocate its self-attention K/V. This should be called
        once at the beginning of an inference session.

        Args:
            batch_size: The batch size for the inference session.
            max_seq_len: The maximum sequence length for the generation.
            encoder_context: The final output of the encoder.
        """
        for i in range(self.num_layers):
            layer = self.layers[f"layer_{i}"]
            layer.init_cache(batch_size, max_seq_len, encoder_context)

    def generate_step(self, 
                      y: jnp.ndarray, 
                      cross_attn_mask: jnp.ndarray | None, 
                      decode_step_index: int):
        """
        Performs a single, cached, autoregressive step through the entire decoder stack.

        This method is called repeatedly in a loop by the top-level generate function.
        It orchestrates the `generate_step` calls for each layer.

        Args:
            y: The input tensor for the single current token. Shape: (batch, 1, d_model).
            cross_attn_mask: The padding mask for the source sequence.
            decode_step_index: The current time-step in the generation loop.

        Returns:
            The output tensor for the single token. Shape: (batch, 1, d_model).
        """
        # Sequentially pass the single token through each layer's generate_step method.
        # This guarantees the layers are processed in the correct order (0, 1, 2, ...).
        for i in range(self.num_layers):
            layer = self.layers[f"layer_{i}"]
            y = layer.generate_step(
                y=y,
                cross_attn_mask=cross_attn_mask,
                decode_step_index=decode_step_index
            )
        
        # Apply the final layer normalization to the stack's single-token output.
        return self.norm(y)

    def __call__(self, 
                 y: jnp.ndarray, 
                 encoder_context: jnp.ndarray, 
                 self_attn_mask: jnp.ndarray | None, 
                 cross_attn_mask: jnp.ndarray | None, 
                 training: bool = False):
        """
        Performs the forward pass for the entire decoder stack.

        Args:
            y: The input tensor for the decoder, typically the target token
               embeddings + positional encodings.
               Shape: (batch_size, target_seq_len, d_model).
            encoder_context: The final output from the encoder stack.
                             Shape: (batch_size, source_seq_len, d_model).
            self_attn_mask: The combined causal and padding mask for the decoder's
                            self-attention.
            cross_attn_mask: The padding mask for the encoder's output, used in
                             cross-attention.
            training: A boolean indicating if the model is in training mode.

        Returns:
            The decoder's final output tensor, ready for projection to logits.
            Shape: (batch_size, target_seq_len, d_model).
        """
        # Sequentially pass the input through each layer in the stack.
        # This loop guarantees the layers are processed in the correct order (0, 1, 2, ...).
        for i in range(self.num_layers):
            layer = self.layers[f"layer_{i}"]
            # The output of one layer becomes the input 'y' for the next.
            y = layer(
                y=y, 
                encoder_context=encoder_context, 
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                training=training
            )
        
        # Apply the final layer normalization to the stack's output.
        return self.norm(y)
    

class PositionalEncoding(nnx.Module):
    """
    Implements the sinusoidal positional encoding module.

    The encodings are not learned parameters but are pre-computed once and
    stored as a standard JAX array attribute on the module.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        *,
        rngs: nnx.Rngs,  # NNX Modules require rngs for initialization
    ):
        """
        Args:
            d_model: The dimensionality of the embedding vectors.
            max_len: The maximum possible sequence length to pre-compute for.
            rngs: Flax NNX Rngs object (not used here but required by the API).
        """
        # Create a (max_len, d_model) matrix to store the encodings.
        # This is a regular JAX array, not an nnx.Param, because it's fixed.
        pe = jnp.zeros((max_len, d_model))

        # Create a vector of positions [0, 1, ..., max_len-1]
        position = jnp.arange(0, max_len, dtype=jnp.float32).reshape(-1, 1)

        # Create the denominator term for the frequencies.
        # The formula is 1 / 10000^(2i/d_model).
        # We compute this in log-space for numerical stability.
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model)
        )

        # Calculate the sines for even indices
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))

        # Calculate the cosines for odd indices
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        # Add a batch dimension so it can be easily added to the input embeddings
        # which have shape (batch_size, seq_len, d_model).
        # Final shape of self.pe will be (1, max_len, d_model).
        self.pe = pe[jnp.newaxis, ...]

    def __call__(self, x: jnp.ndarray, start_index: int = 0) -> jnp.ndarray:
        """
        Adds positional encodings to the input embeddings.

        Args:
            x: The input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            The input embeddings with added positional information, same shape.
        """
        # Get the sequence length from the input tensor
        seq_len = x.shape[1]

        # Add the pre-computed positional encodings to the input embeddings.
        # We slice self.pe to match the input's sequence length.
        # JAX's broadcasting handles the addition across the batch dimension.
        # x shape: (batch_size, seq_len, d_model)
        # self.pe[:, :seq_len, :] shape: (1, seq_len, d_model)
        #output = x + self.pe[:, :seq_len, :]
        
        # Slice the positional encoding matrix starting from the given index.
        pos_encoding_slice = self.pe[:, start_index: start_index + seq_len, :]
        return x + pos_encoding_slice
