"""
    This module contains custom Flax transformers for machine learning tasks.
"""

import jax.numpy as jnp
from flax import nnx
from krk_ml_utils.transformer_components import PositionalEncoding, TransformerEncoder, TransformerDecoder


# Helper function for creating masks. 
def create_padding_mask(tokens: jnp.ndarray, pad_token_id: int = 250002) -> jnp.ndarray:
    """Creates a padding mask from a batch of token IDs."""
    # Mask is 1 where token is NOT padding, 0 where it is.
    mask = (tokens != pad_token_id)
    # Add dimensions for multi-head attention compatibility: (batch, 1, 1, seq_len)
    return mask[:, jnp.newaxis, jnp.newaxis, :]

def create_causal_mask(seq_len: int):
    """Creates a causal (look-ahead) mask of shape (1, 1, seq_len, seq_len)."""
    #mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[jnp.newaxis, jnp.newaxis, :, :]

class Vanilla_Transformer_v1(nnx.Module):
    """A simple language model transformer using Flax."""
    
    def _initFunc(self):
        # Initialize embedding
        self.embedding = nnx.Embed(
             num_embeddings=self.vocab_size,
             features=self.d_model,
             rngs=self.rngs
             )
        
        # Initialize positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length, rngs=self.rngs)
        
        # Initialize dropout layer
        self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=self.rngs)
        
        # Initialize the encoder
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            num_layers=self.num_layers_enc,
            num_heads=self.num_heads_enc,
            d_ff=self.d_dff_enc,
            rngs=self.rngs,
            dropout_rate=self.dropout_rate
        )
        
        # Initialize the decoder
        self.decoder = TransformerDecoder(
            num_layers=self.num_layers_dec,
            d_model=self.d_model,
            num_heads=self.num_heads_dec,
            d_ff=self.d_dff_dec,
            rngs=self.rngs,
            dropout_rate=self.dropout_rate
        )
        
        
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int, num_layers_enc: int, num_layers_dec: int, num_heads_enc: int, num_heads_dec: int, d_dff_enc: int, d_dff_dec: int, seed: int = 42, dropout_rate: float = 0.1):
        super().__init__()
        
        # Initialize model parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.num_heads_enc = num_heads_enc
        self.num_heads_dec = num_heads_dec
        self.d_dff_enc = d_dff_enc
        self.d_dff_dec = d_dff_dec
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rngs = nnx.Rngs(self.seed)
        self._initFunc()

    def __call__(self, source_tokens: jnp.ndarray, target_tokens: jnp.ndarray, training: bool = False, pad_token_id: int = 250002) -> jnp.ndarray:
        """
        Performs the full forward pass for training.

        Args:
            source_tokens: The input token IDs for the encoder.
                           Shape: (batch_size, source_seq_len).
            target_tokens: The input token IDs for the decoder (e.g., the shifted-right target).
                           Shape: (batch_size, target_seq_len).
            training: A boolean indicating if the model is in training mode.

        Returns:
            The final output logits over the vocabulary.
            Shape: (batch_size, target_seq_len, vocab_size).
        """
        
        if pad_token_id is None:
            pad_token_id = -1  # Default padding token ID if not provided
            
        # 1. Create the padding mask for the encoder's input.
        # This prevents attention from being paid to padding tokens.
        
        source_mask = create_padding_mask(source_tokens, pad_token_id=pad_token_id)
        target_padding_mask = create_padding_mask(target_tokens, pad_token_id=pad_token_id)
        target_causal_mask = create_causal_mask(target_tokens.shape[1])
        # Combine the masks for the decoder
        target_mask = target_padding_mask & target_causal_mask

        # 2. Process the source sequence through the encoder.
        # (batch, seq_len) -> (batch, seq_len, d_model)
        source_emb = self.embedding(source_tokens)
        # Apply embedding scaling as per "Attention is All You Need"
        source_emb = source_emb * jnp.sqrt(self.d_model)
        source_emb = self.pos_encoding(source_emb)
        source_emb = self.dropout(source_emb, deterministic=not training)
        
        # The encoder processes the embeddings to create a rich contextual representation.
        encoder_context = self.encoder(source_emb, mask=source_mask, training=training)

        target_embed = self.embedding(target_tokens)
        # Apply embedding scaling as per "Attention is All You Need"
        target_embed = target_embed * jnp.sqrt(self.d_model)
        target_embed = self.pos_encoding(target_embed)
        target_embed = self.dropout(target_embed, deterministic=not training)
        
        decoder_output = self.decoder(
            y=target_embed,
            encoder_context=encoder_context,
            self_attn_mask=target_mask,
            cross_attn_mask=source_mask,
            training=training
        )
        
        # 4. Final Projection to Logits (Weight Tying)
        # This is the final step. The output from the decoder is projected into the
        # vocabulary space to get a score for each possible next token.
        # We use the transpose of the embedding matrix to do this, tying the weights.
        logits = decoder_output @ self.embedding.embedding.T
        return logits
    
    def generate(self, source_tokens: jnp.ndarray, start_token_id: int, max_generate_len: int, eos_token_id: int, pad_token_id: int = -1):
        """
        Generates a sequence of tokens autoregressively using KV caching.

        Args:
            source_tokens: The input token IDs for the encoder. Shape: (batch_size, source_seq_len).
            start_token_id: The token ID to begin generation (e.g., BOS token).
            max_generate_len: The maximum length of the sequence to be generated.
            eos_token_id: The token ID that signifies the end of a sequence.
            pad_token_id: The ID for padding tokens, used to create the source mask.

        Returns:
            The generated sequence of token IDs. Shape: (batch_size, generated_len).
        """
        batch_size = source_tokens.shape[0]

        # --- Phase 1: One-Time Setup ---

        # 1a. Create the padding mask for the encoder's cross-attention.
        source_mask = create_padding_mask(source_tokens, pad_token_id)

        # 1b. Run the encoder pass once.
        source_emb = self.embedding(source_tokens)
        # Apply embedding scaling as per "Attention is All You Need"
        source_emb = source_emb * jnp.sqrt(self.d_model)
        source_emb = self.pos_encoding(source_emb, start_index=0)
        encoder_context = self.encoder(source_emb, mask=source_mask, training=False)

        # 1c. Initialize the decoder's caches. This pre-computes cross-attention
        # K/V and pre-allocates memory for self-attention K/V.
        self.decoder.init_cache(batch_size, max_generate_len, encoder_context)
        
        # 1d. Prepare the initial state for the generation loop.
        # Start with the `start_token_id` for each sequence in the batch.
        generated_tokens = jnp.full((batch_size, 1), start_token_id, dtype=jnp.int32)
        
        # --- Phase 2: The Autoregressive Loop ---

        for t in range(max_generate_len - 1):
            # The input for this step is the single last token generated.
            last_token = generated_tokens[:, -1]

            # Embed the token and apply the positional encoding for the current time-step 't'.
            token_emb = self.embedding(last_token)
            # Apply embedding scaling as per "Attention is All You Need"
            token_emb = token_emb * jnp.sqrt(self.d_model)
            # Add a sequence dimension of 1 to make it (batch, 1, d_model)
            token_emb = token_emb[:, jnp.newaxis, :]
            token_emb_pos = self.pos_encoding(token_emb, start_index=t)

            # Call the decoder's cached generation step.
            decoder_output = self.decoder.generate_step(
                y=token_emb_pos,
                cross_attn_mask=source_mask,
                decode_step_index=t
            )

            # Project the single output vector to logits over the vocabulary.
            logits = decoder_output @ self.embedding.embedding.T

            # Sample the next token using greedy decoding (taking the most likely token).
            next_token = jnp.argmax(logits, axis=-1)

            # Append the new token to our sequence of generated tokens.
            generated_tokens = jnp.concatenate([generated_tokens, next_token], axis=1)

            # Early stopping: if all sequences in the batch have generated an
            # end-of-sequence token, we can stop generation.
            if jnp.all(jnp.any(generated_tokens == eos_token_id, axis=1)):
                break
                
        return generated_tokens
