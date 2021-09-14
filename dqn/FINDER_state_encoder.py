import tensorflow as tf
from dqn.attention_module import MultiHeadAttention, create_padding_mask

class BasicStateEncoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, state_embed_dim, node_embed_dim):
        super(BasicStateEncoder, self).__init__()
        self.state_embed_dim = state_embed_dim
        self.node_embed_dim = node_embed_dim
        
        self.start_param = tf.cast(placeholder_dict['start_param'], tf.float32)
        self.end_param = tf.cast(placeholder_dict['end_param'], tf.float32)
        self.subgsum_param = tf.cast(placeholder_dict['subgsum_param'], tf.float32)

    def call(self, node_embed):
        start_node_embed = tf.sparse.sparse_dense_matmul(self.start_param, node_embed)
        end_node_embed = tf.sparse.sparse_dense_matmul(self.end_param, node_embed) 
        
        aggregated_node_embed = tf.sparse.sparse_dense_matmul(self.subgsum_param, node_embed)
        aggregated_node_embed = tf.reshape(aggregated_node_embed, [-1, self.node_embed_dim])
        
        raw_state_embed = tf.concat([aggregated_node_embed, start_node_embed, end_node_embed], axis=1)
        state_embed = tf.reshape(raw_state_embed, [-1, 3*self.node_embed_dim])
        return state_embed

class MHAStateEncoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, state_embed_dim, node_embed_dim, num_heads=8, d_model=128, max_nodes=20, rate=0.1):
        super(MHAStateEncoder, self).__init__()
        self.max_nodes = max_nodes
        self.state_embed_dim = state_embed_dim
        self.node_embed_dim = node_embed_dim
        # access relevant placeholders
        self.start_param = tf.cast(placeholder_dict['start_param'], tf.float32)
        self.end_param = tf.cast(placeholder_dict['end_param'], tf.float32)
        self.agg_state_param = tf.cast(placeholder_dict['agg_state_param'], tf.float32)
        # self.subgsum_param = tf.cast(placeholder_dict['subgsum_param'], tf.float32)
        self.pad_node_param = tf.cast(placeholder_dict['pad_node_param'], tf.float32)
        # define layers
        self.dropout = tf.keras.layers.Dropout(rate)
        self.mha = MultiHeadAttention(d_model, num_heads=num_heads, out_put_dim=state_embed_dim)
        
        # self.dense = tf.keras.layers.Dense(state_embed_dim, use_bias=True, activation='relu')
        # self.norm = tf.keras.layers.LayerNormalization(axis=2)
        
    def call(self, node_embed, training):
        start_node_embed = tf.sparse.sparse_dense_matmul(self.start_param, node_embed)
        end_node_embed = tf.sparse.sparse_dense_matmul(self.end_param, node_embed) 
        # replace with learned weights in case of t=0

        # aggregated_state_embed = tf.sparse.sparse_dense_matmul(self.subgsum_param, node_embed)
        aggregated_node_embed = tf.sparse.sparse_dense_matmul(self.agg_state_param, node_embed)
        aggregated_node_embed = tf.reshape(aggregated_node_embed, [-1, self.node_embed_dim])
        
        raw_state_embed = tf.concat([aggregated_node_embed, start_node_embed, end_node_embed], axis=1)
        raw_state_embed = tf.reshape(raw_state_embed, [-1, 3*self.node_embed_dim])
        raw_state_embed = self.dropout(raw_state_embed, training=training)
        
        # project the raw embedding using a nonlinear dense layer such that it has the desired state dimension 
        # (optional, can be absored into mha layer)
        # proj_state_embed = self.dense(raw_state_embed)
        # proj_state_embed = tf.reshape(proj_state_embed, [-1, 1, self.state_embed_dim])
        
        # here apply some matrix to set all embeddings of irrelevant nodes to zero and pad the input
        node_embed = tf.sparse.sparse_dense_matmul(self.pad_node_param, node_embed)
        node_embed = tf.reshape(node_embed, [-1, self.max_nodes, self.node_embed_dim])
        # potentially apply one layer of masked self attention (increases complexity)
        
        # refine context
        # at best mask all nodes that have already been selected (apart from last and start node)
        mask_input = tf.reduce_sum(node_embed, -1)
        padding_mask = create_padding_mask(mask_input)
        final_state_embed, attention_weights = self.mha(raw_state_embed, node_embed, node_embed, mask=padding_mask)
        
        # norm and add to prevent vanishing gradients (optional)
        # final_state_embed = self.norm(tf.math.add(final_state_embed, proj_state_embed))
        final_state_embed = tf.reshape(final_state_embed, [-1, self.state_embed_dim])
        return final_state_embed
    
# non functional
class RNNStateEncoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, state_embed_dim, node_embed_dim, num_heads=8, d_model=128, max_nodes=20, rate=0.1):
        super(RNNStateEncoder, self).__init__()
        pass

    def call(self, node_embed):
        # num_samples = tf.shape(self.placeholder_dict['subgsum_param'])[0]
        # covered_embed_padded = tf.sparse.sparse_dense_matmul(tf.cast(self.placeholder_dict['state_param'], tf.float32), cur_node_embed)
        # covered_embed_padded_reshaped = tf.reshape(covered_embed_padded, [num_samples, -1, self.cfg['node_embed_dim']])
        # masked_covered_embed = tf.keras.layers.Masking(mask_value=0.)(covered_embed_padded_reshaped)

        # RNN_cell = tf.keras.layers.LSTMCell(units=64)
        # RNN_layer = tf.keras.layers.RNN(cell=RNN_cell, return_state=True, return_sequences=True)

        # whole_seq_output, final_memory_state, final_carry_state = RNN_layer(masked_covered_embed)
        # cur_state_embed = final_carry_state
        pass