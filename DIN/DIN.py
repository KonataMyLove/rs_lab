import tensorflow as tf

class Attention_Layer:
    def __init__(self, att_hidden_units=(80, 40), activation='sigmoid'):
        super(Attention_Layer, self).__init__()
        self.att_hidden_units = att_hidden_units
    
    def call(self, inputs):
        """
        [item_embed, seq_embed, seq_embed, mask]
        item_embed: 候选商品，即目标 (None, embedding_dim)
        seq_embed: 用户历史商品序列 (None, max_len, embedding_dim) max_len是最大历史序列长度
        mask: 指示seq_embed的padding情况
        """
        with tf.variable_scope("din_attn", reuse=tf.AUTO_REUSE):
            q, k, v, key_mask = inputs
            q = tf.expand_dims(q, axis=1)
            q = tf.tile(q, [1, k.shape[1], 1])  # (None, max_len, embedding_dim)
            
            info = tf.concat([q, k, q - k, q * k], axis=-1)  # 进行attention分数计算的两块，历史序列和候选商品
            for unit in self.att_hidden_units:
                info = tf.layers.dense(info,
                                       unit,
                                       activation=tf.nn.sigmoid)
            
            outputs = tf.layers.dense(info, 1)  # (None, max_len, 1)
            outputs = tf.squeeze(outputs, axis=-1)  # 注意力分数的logits

            # mask
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_mask, 0), paddings, outputs)

            # softmax分数计算
            attn_score = tf.nn.softmax(outputs, axis=-1)  # (None, max_len)
            attn_score = tf.expand_dims(attn_score, axis=1)
            weighted_v = tf.matmul(attn_score, v)  # (None, 1, embedding_dim)
            weighted_v = tf.squeeze(weighted_v, axis=1)

            return weighted_v

class Dice:
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = tf.get_variable(name='alpha', shape=(), dtype=tf.float32)

    def call(self, x):
        with tf.variable_scope("dice", reuse=tf.AUTO_REUSE):
            x_normed = tf.layers.batch_normalization(x, training=True)  # 注意推理时tf1 batchnorm的用法
            x_p = tf.nn.sigmoid(x_normed)
            output = self.alpha * (1.0 - x_p) * x + x_p * x

        return output

class DIN:
    def __init__(self, feature_columns, behavior_feature_list, maxlen=40, dropout=0.5,
                 att_hidden_units=(80, 40), att_activation='sigmoid', fnn_hidden_units=(80, 40)):
        self.maxlen = maxlen
        
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns           # 这里把连续特征和离散特征分别取出来， 因为后面两者的处理方式不同
        
        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)      # 这个other_sparse就是离散特征中去掉了能表示用户行为的特征列
        self.dense_len = len(self.dense_feature_columns)    
        self.behavior_num = len(behavior_feature_list)

        self.fnn_hidden_units = fnn_hidden_units

        # embedding层， 这里分为两部分的embedding， 第一部分是普通的离散特征， 第二部分是能表示用户历史行为的离散特征， 这一块后面要进注意力和当前的商品计算相关性
        # 普通离散特征embedding，用于与dense特征拼接
        self.embed_sparse_layers = [tf.get_variable(name="embedding_1",
                                                    shape=(feat['feat_num'], feat['embed_dim']),
                                                    initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                                    regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)
                                                   ) for feat in self.sparse_feature_columns if feat['feat'] not in behavior_feature_list]

        self.embed_seq_layers = [tf.get_variable(name="embedding_2",
                                                 shape=(feat['feat_num'], feat['embed_dim']),
                                                 initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                                 regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)
                                                ) for feat in self.sparse_feature_columns if feat['feat'] in behavior_feature_list]

        # DIN注意力层
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.dropout = dropout

    def call(self, inputs):
        """
        inputs: [dense_input, sparse_input, seq_input, item_input]
        dense_input： 连续型的特征输入， 维度是(None, dense_len)
        sparse_input: 离散型的特征输入， 维度是(None, other_sparse_len)
        seq_inputs: 用户的历史行为序列(None, maxlen, behavior_len) behavior_len即行为序列的特征种类，此处默认为1即可
        item_inputs： 当前的候选商品序列 (None, behavior_len)
        """
        with tf.variable_scope("din", reuse=tf.AUTO_REUSE):
            dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs

            # 生成用户历史行为序列mask
            mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)

            # 连续特征和非用户行为序列离散特征拼接
            other_info = dense_inputs
            for i in range(self.other_sparse_len):
                other_info = tf.concat([other_info,
                                        tf.nn.embedding_lookup(self.embed_sparse_layers[i], sparse_inputs[:, i])
                                       ], axis=-1)

            seq_embed = tf.concat([tf.nn.embedding_lookup(self.embed_seq_layers[i], seq_inputs[:, :, i])
                                   for i in range(self.behavior_num)], axis=-1)   # [None, max_len, embed_dim]
            item_embed = tf.concat([tf.nn.embedding_lookup(self.embed_seq_layers[i], item_inputs[:, i])
                                    for i in range(self.behavior_num)], axis=-1)  # [None, embed_dim]

            # attention计算，返回的是已经注意力加权后的用户历史序列特征，即用户特征向量
            user_info = self.attention_layer.call((item_embed, seq_embed, seq_embed, mask))  # (None, embed_dim)

            if self.dense_len > 0 or self.other_sparse_len > 0:
                info_all = tf.concat([user_info, item_embed, other_info], axis=-1)   # (None, dense_len + other_sparse_len + embed_dim+embed_dim)  
            else:
                info_all = tf.concat([user_info, item_embed], axis=-1)
            
            info_all = tf.layers.batch_normalization(info_all)

            # ffn
            for unit in self.fnn_hidden_units:
                info_all = tf.layers.dense(info_all,
                                           unit,
                                           activation=tf.nn.relu)
            info_all = tf.nn.dropout(info_all, keep_prob=self.dropout)
            outputs = tf.nn.sigmoid(tf.layers.dense(info_all, 1))

            return outputs



# if __name__ == '__main__':
#     # din_attn = Attention_Layer()
#     # dice = Dice()
#     feature_columns = ([{'feat': 'age'}],
#                        [{'feat': 'cate', 'feat_num': 10, 'embed_dim': 80}, {'feat': 'session', 'feat_num': 100, 'embed_dim': 80}])
#     behavior_feature_list = ['session']
#     din = DIN(feature_columns, behavior_feature_list)
#     dense_input = tf.ones([1, 8])
#     sparse_input = tf.cast(tf.ones([1, 1]), dtype=tf.int32)
#     seq_input = tf.cast(tf.ones([1, 40, 1]), dtype=tf.int32)
#     item_input = tf.cast(tf.ones([1, 1]), dtype=tf.int32)
#     inputs = (dense_input, sparse_input, seq_input, item_input)
#     with tf.Session() as sess:
#         # res = din_attn.call((item_embed, seq_embed, seq_embed, mask))
#         res = din.call(inputs)
#         sess.run(tf.global_variables_initializer())  # 初始化要放在所有网络参数都定义之后
#         print(sess.run(res))  # sess.run才开始真正运行