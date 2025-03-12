import tensorflow as tf

class Attention_Layer:
    def __init__(self, att_hidden_units=(80, 40), activation='sigmoid'):
        super(Attention_Layer, self).__init__()
        self.att_hidden_units = att_hidden_units
    
    def attn(self, inputs):
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
            outputs = tf.where(tf.equal(key_mask, False), paddings, outputs)

            # softmax分数计算
            attn_score = tf.nn.softmax(outputs, axis=-1)  # (None, max_len)
            attn_score = tf.expand_dims(attn_score, axis=1)
            weighted_v = tf.matmul(attn_score, v)  # (None, 1, embedding_dim)
            weighted_v = tf.squeeze(weighted_v, axis=1)

            return weighted_v

if __name__ == '__main__':
    din_attn = Attention_Layer()
    item_embed = tf.zeros(shape=[1, 8])
    seq_embed = tf.zeros(shape=[1, 40, 8])
    mask = tf.constant(True, shape=[1, 40], dtype=tf.bool)
    with tf.Session() as sess:
        res = din_attn.attn((item_embed, seq_embed, seq_embed, mask))
        sess.run(tf.global_variables_initializer())  # 初始化要放在所有网络参数都定义之后
        print(sess.run(res))  # sess.run才开始真正运行