import tensorflow as tf
import numpy as np

# FM特征交叉
class CrossLayer:
    def __init__(self, input_dim, output_dim=10):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 交叉特征权重
        self.kernel = tf.get_variable(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer=tf.keras.initializers.glorot_uniform(),
                                      trainable=True)
    
    def call(self, x):
        a = tf.square(tf.matmul(x, self.kernel))
        b = tf.matmul(tf.square(x), tf.square(self.kernel))
        return 0.5 * tf.reduce_mean(a - b, axis=1, keepdims=True)

def eval(pred, labels):
    with tf.variable_scope("eval", reuse=tf.AUTO_REUSE):
        pred = tf.nn.sigmoid(pred)
        mask = tf.greater(pred, 0.8)
        pred_binary = tf.cast(mask, dtype=tf.float32)  # (bsz, 1)
        pred_bool = tf.cast(pred_binary, tf.bool)
        labels_bool = tf.cast(tf.cast(labels, tf.int32), tf.bool)
        Precision = tf.reduce_sum(tf.cast(tf.logical_and(pred_bool, labels_bool), tf.float32)) / tf.reduce_sum(pred_binary)
        Recall = tf.reduce_sum(tf.cast(tf.logical_and(pred_bool, labels_bool), tf.float32)) / tf.reduce_sum(labels)

    return Precision, Recall

def FM(feature_dim):
    with tf.variable_scope("fm", reuse=tf.AUTO_REUSE):
        inputs = tf.placeholder(shape=(None, feature_dim),
                                dtype=tf.float32)
        labels = tf.placeholder(shape=(None, 1),
                                dtype=tf.float32)
        linear_feats = tf.layers.dense(inputs,
                                       1,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                       bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)
                                       )  # 一阶特征 (bsz, 1)
        cross_layer = CrossLayer(feature_dim)  # 定义特征交叉层
        cross_feats = cross_layer.call(inputs)  # 二阶特征 (bsz, 1)
        pred = tf.add(linear_feats, cross_feats)
        # pred = tf.nn.sigmoid(info)
        Pre, Rec = eval(pred, labels)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=pred)
        loss = tf.reduce_mean(loss)

    return inputs, labels, loss, Pre, Rec

# if __name__ == '__main__':
#     inputs_1 = np.ones([1, 32])
#     with tf.Session() as sess:
#         inputs_ph = FM(32)
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(pred, feed_dict={inputs_ph: inputs_1}))
