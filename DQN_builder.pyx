import tensorflow as tf
import numpy as np

class DQN_builder:

    def __init__(self):
        pass
    """
    def BuildNet(self):
        # initialize weight matrices
        # w_n2l, w_e2l, 

        
        ############# get embeddings
        # [node_cnt, node_dim] * [node_dim, embed_dim] = [node_cnt, embed_dim], no sparse
        node_init = tf.matmul(tf.cast(self.node_input, tf.float32), w_n2l)
        cur_node_embed = tf.nn.relu(node_init)
        cur_node_embed = tf.nn.l2_normalize(cur_node_embed, axis=1)

        # [edge_cnt, edge_dim] * [edge_dim, embed_dim] = [edge_cnt, embed_dim]
        edge_init = tf.matmul(tf.cast(self.edge_input, tf.float32), w_e2l)
        cur_edge_embed = tf.nn.relu(edge_init)
        cur_edge_embed = tf.nn.l2_normalize(cur_edge_embed, axis=1)

        lv = 0
        while lv < max_bp_iter:
            cur_node_embed_prev = cur_node_embed
            lv = lv + 1
            ###################### update edges ####################################
            # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim]
            msg_linear_node = tf.matmul(cur_node_embed, p_node_conv1)
            # [edge_cnt, node_cnt] * [node_cnt, embed_dim] = [edge_cnt, embed_dim]
            n2e = tf.sparse_tensor_dense_matmul(tf.cast(self.n2esum_param, tf.float32), msg_linear_node)
            # [edge_cnt, embed_dim] + [edge_cnt, embed_dim] = [edge_cnt, embed_dim]
            # n2e_linear = tf.add(n2e, edge_init)
            # n2e_linear = tf.concat([tf.matmul(n2e, trans_edge_1), tf.matmul(edge_init, trans_edge_2)], axis=1)    #we used
            n2e_linear = tf.concat([tf.matmul(n2e, trans_edge_1), tf.matmul(cur_edge_embed, trans_edge_2)], axis=1)    #we used
            # n2e_linear = tf.concat([n2e, edge_init], axis=1)    # [edge_cnt, 2*embed_dim]
            # [edge_cnt, embed_dim]
            cur_edge_embed = tf.nn.relu(n2e_linear)
            ### if MLP
            # cur_edge_embed_temp = tf.nn.relu(tf.matmul(n2e_linear, trans_edge_1))   #[edge_cnt, embed_dim]
            # cur_edge_embed = tf.nn.relu(tf.matmul(cur_edge_embed_temp, trans_edge_2))   #[edge_cnt, embed_dim]
            cur_edge_embed = tf.nn.l2_normalize(cur_edge_embed, axis=1)

            ###################### update nodes ####################################
            # msg_linear_edge = tf.matmul(cur_edge_embed, p_node_conv2)
            # [node_cnt, edge_cnt] * [edge_cnt, embed_dim] = [node_cnt, embed_dim]
            e2n = tf.sparse_tensor_dense_matmul(tf.cast(self.e2nsum_param, tf.float32), cur_edge_embed)
            # [node_cnt, embed_dim] * [embed_dim, embed_dim] + [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim]
            # node_linear = tf.add(tf.matmul(e2n, trans_node_1), tf.matmul(cur_node_embed, trans_node_2))
            node_linear = tf.concat([tf.matmul(e2n, trans_node_1), tf.matmul(cur_node_embed, trans_node_2)], axis=1)    #we used
            # node_linear = tf.concat([e2n, cur_node_embed], axis=1)  #[node_cnt, 2*embed_dim]
            # [node_cnt, embed_dim]
            cur_node_embed = tf.nn.relu(node_linear)
            ## if MLP
            # cur_node_embed_temp = tf.nn.relu(tf.matmul(node_linear, trans_node_1))  # [node_cnt, embed_dim]
            # cur_node_embed = tf.nn.relu(tf.matmul(cur_node_embed_temp, trans_node_2))   # [node_cnt, embed_dim]
            cur_node_embed = tf.nn.l2_normalize(cur_node_embed, axis=1)
            cur_node_embed = tf.matmul(tf.concat([cur_node_embed, cur_node_embed_prev], axis=1), w_l)
    """