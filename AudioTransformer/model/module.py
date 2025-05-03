import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, trainable_embed=False, **kwargs):
        super().__init__(**kwargs)
        
        # Generate positional encoding matrix
        position_embedding_matrix = self.get_position_encoding(sequence_length, embed_dim)
        
        # Non-trainable Embedding layer to store positional encodings
        if trainable_embed == False:
            self.position_embedding_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=embed_dim,
                weights=[position_embedding_matrix],
                trainable=False
            )
        else:
            self.position_embedding_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=embed_dim,
                trainable=True
            )
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
 
    def call(self, inputs):        
        # Get position indices (0, 1, ..., sequence_length-1)
        sequence_length = tf.shape(inputs)[1]  # Ambil sequence_length dari inputs
        position_indices = tf.range(sequence_length)  # Shape: (sequence_length,)
        
        # Get positional encoding
        positional_encoding = self.position_embedding_layer(position_indices)  # Shape: (sequence_length, embed_dim)
        
        # Expand dimensions for broadcasting: (1, sequence_length, embed_dim)
        return positional_encoding[tf.newaxis, :, :]
