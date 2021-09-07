import tensorflow as tf
import numpy as np




class AGNNEncoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, num_layers, node_embed_dim, weight_stddev, rate=0.1):
        super(AGNNEncoder, self).__init__()
        pass
    def call(self, x):
        pass

class AGNNEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, num_layers, node_embed_dim, weight_stddev, rate=0.1):
        super(AGNNEncoderLayer, self).__init__()
        pass
    def call(self, x):
        pass

class MLPdecoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, state_embed_dim, node_embed_dim, weight_stddev, hidden_dim=32, rate=0.1):
        super(MLPdecoder, self).__init__()
        self.h1_weight = tf.Variable(tf.truncated_normal([state_embed_dim + node_embed_dim, hidden_dim], stddev=weight_stddev), tf.float32)
        self.last_w = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=weight_stddev), tf.float32)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.rep_global = tf.cast(placeholder_dict['rep_global'], tf.float32)
    
    def call(self, node_embed, state_embed, training, repeat_states):
        if repeat_states:
            state_embed = tf.sparse.sparse_dense_matmul(self.rep_global, state_embed)

        embed_s_a = tf.concat([node_embed, state_embed], axis=1)
        embed_s_a = self.dropout(embed_s_a, training=training)
        # [batch_size, (2)node_embed_dim] * [(2)node_embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
        hidden = tf.matmul(embed_s_a, self.h1_weight)
        # [batch_size, reg_hidden]
        last_output = tf.nn.relu(hidden)

         # [batch_size, reg_hidden] * [reg_hidden, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, self.last_w)

        return q_pred

class MHAdecoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, node_embed_dim, state_embed_dim, d_model=128, num_heads_1=8, num_heads_2=1, max_nodes=20):
        super(MHAdecoder, self).__init__()
        self.max_nodes = max_nodes
        self.state_embed_dim = state_embed_dim
        self.node_embed_dim = node_embed_dim
        
        self.rep_global = tf.cast(placeholder_dict['rep_global'], tf.float32)

        self.mha1 = MultiHeadAttention(d_model, num_heads=num_heads_1)
        self.mha2 = MultiHeadAttention(d_model, num_heads=num_heads_2)

    def call(self, node_embed, state_embed, training, repeat_states):
        if repeat_states:
            state_embed = tf.sparse.sparse_dense_matmul(self.rep_global, state_embed)
        state_embed = tf.reshape(state_embed, (-1, 1, self.state_embed_dim))
        node_embed = tf.reshape(node_embed, (-1, self.max_nodes, self.node_embed_dim))
        dec_padding_mask = self.create_mask(node_embed)
        output, attention_weights = self.mha1(q=state_embed, k=node_embed, v=node_embed, mask=None)
        return output

    def create_mask(self, inp):
        # Decoder padding mask
        dec_padding_mask = create_padding_mask(inp)
        return dec_padding_mask

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False)

        self.dense = tf.keras.layers.Dense(d_model, use_bias=False)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)