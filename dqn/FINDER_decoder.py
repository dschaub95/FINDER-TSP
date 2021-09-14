import tensorflow as tf
from dqn.attention_module import MultiHeadAttention, create_padding_mask

class MLPdecoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, state_embed_dim, node_embed_dim, weight_stddev, hidden_dim=32, rate=0.1):
        super(MLPdecoder, self).__init__()
        self.h1_weight = tf.Variable(tf.truncated_normal([state_embed_dim + node_embed_dim, hidden_dim], stddev=weight_stddev), tf.float32)
        self.last_w = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=weight_stddev), tf.float32)
        self.dropout = tf.keras.layers.Dropout(rate)
        # access relevant placeholders
        self.rep_global = tf.cast(placeholder_dict['rep_global'], tf.float32)
        # self.action_select = tf.cast(self.placeholder_dict['action_select'], tf.float32)
    
    def call(self, state_embed, action_embed, q_on_all, training):
        if q_on_all:
            state_embed = tf.sparse.sparse_dense_matmul(self.rep_global, state_embed)    
        embed_s_a = tf.concat([state_embed, action_embed], axis=1)
        embed_s_a = self.dropout(embed_s_a, training=training)
        # [batch_size, (2)node_embed_dim] * [(2)node_embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
        hidden = tf.matmul(embed_s_a, self.h1_weight)
        # [batch_size, reg_hidden]
        last_output = tf.nn.relu(hidden)

         # [batch_size, reg_hidden] * [reg_hidden, 1] = [batch_size, 1]
        q_values = tf.matmul(last_output, self.last_w)
        q_values = tf.reshape(q_values, [-1, 1])
        return q_values


class AttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, state_embed_dim, node_embed_dim, max_nodes=20, d_model=128, use_biases=False, rate=0.1):
        super(AttentionDecoder, self).__init__()
        self.state_embed_dim = state_embed_dim
        self.node_embed_dim = node_embed_dim
        self.max_nodes = max_nodes

        self.action_dropout = tf.keras.layers.Dropout(rate)
        self.state_dropout = tf.keras.layers.Dropout(rate)
        self.wq = tf.keras.layers.Dense(d_model, use_bias=use_biases)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=use_biases)

        # self.rep_global = tf.cast(placeholder_dict['rep_global'], tf.float32)
        self.pad_node_param = tf.cast(placeholder_dict['pad_node_param'], tf.float32)
        self.pad_reverse_param = tf.cast(placeholder_dict['pad_reverse_param'], tf.float32)
        self.mask_param = tf.cast(placeholder_dict['mask_param'], tf.float32)
    
    def call(self, state_embed, action_embed, q_on_all, training): 
        if q_on_all:
            action_embed = tf.sparse.sparse_dense_matmul(self.pad_node_param, action_embed)
            action_embed = tf.reshape(action_embed, [-1, self.max_nodes, self.node_embed_dim])
        else:
            action_embed = tf.reshape(action_embed, [-1, 1, self.node_embed_dim])
        action_embed = self.action_dropout(action_embed, training=training)
        # state_embed = tf.sparse.sparse_dense_matmul(self.pad_node_param, state_embed)
        state_embed = tf.reshape(state_embed, [-1, 1, self.state_embed_dim])
        state_embed = self.state_dropout(state_embed, training=training)

        q = self.wq(state_embed)  # (batch_size, seq_len, d_model)
        k = self.wk(action_embed)  # (batch_size, seq_len, d_model)

        q_values = tf.matmul(q, k, transpose_b=True)
        q_values = tf.reshape(q_values, [-1, 1])
        # make backtransfo from padded to node_cnt
        if q_on_all:
            q_values = tf.sparse.sparse_dense_matmul(self.pad_reverse_param, q_values)

        # apply mask to make loss independent of selected nodes q-values, not used during training since only valid actions are considered
        # only works if no nodes are deleted and the graph dimension is fixed during training and testing accross batches
        # if mask_qvalues:
        #     q_values = tf.sparse.sparse_dense_matmul(self.mask_param, q_values)
        #     mask = tf.cast(tf.math.equal(q_values, 0), tf.float32)
        #     q_values += (mask * -1e9)
        return q_values





# needs refinement 
class MHAdecoder(tf.keras.layers.Layer):
    def __init__(self, placeholder_dict, node_embed_dim, state_embed_dim, d_model=128, num_heads_1=8, num_heads_2=1, max_nodes=20):
        super(MHAdecoder, self).__init__()
        self.max_nodes = max_nodes
        self.state_embed_dim = state_embed_dim
        self.node_embed_dim = node_embed_dim
        self.d_model = d_model

        self.mha1 = MultiHeadAttention(d_model, num_heads=num_heads_1, out_put_dim=d_model)
        self.mha2 = MultiHeadAttention(d_model, num_heads=num_heads_2, out_put_dim=d_model)

        self.dense = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, node_embed, state_embed, training, q_on_all=True):
        # node embed needs to have fixed dimensions for each step
        state_embed = tf.reshape(state_embed, (-1, 1, self.state_embed_dim))
        node_embed = tf.reshape(node_embed, (-1, self.max_nodes, self.node_embed_dim))
        dec_padding_mask = self.create_mask(node_embed)
        # refine context
        # at best mask all nodes that have already been selected (apart from last and start node)
        context_embed, attention_weights_1 = self.mha1(q=state_embed, k=node_embed, v=node_embed, mask=None)
        # duplicate context embeddings to obtain q values over all nodes
        # context_embed = tf.tile(context_embed, multiples=[1,self.max_nodes,1])

        # context_embed = tf.reshape(context_embed, [-1, self.d_model])
        # context_embed = tf.sparse.sparse_dense_matmul(self.rep_global, context_embed)
        
        # depending on whether we only want to predict the q value for a single action (e.g., during training)
        # at best define a mask before the last attention layer otherwise do it afterwards
        q_values_raw, attention_weights_2 = self.mha1(q=context_embed, k=node_embed, v=node_embed, mask=None)
        if not q_on_all:
            pass
        
        # use a final dense layer for scaling to get appropriate q values
        final_q_values = self.dense(q_values_raw)
        return final_q_values
    
    def create_mask(self, inp):
        # Decoder padding mask
        dec_padding_mask = create_padding_mask(inp)
        return dec_padding_mask