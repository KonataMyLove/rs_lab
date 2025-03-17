import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from FM import FM

# dense特征取对数， sparse特征类别编码
def process_feat(data, dense_feats, sparse_feats):
    df = data.copy()
    
    # dense
    df_dense = df[dense_feats].fillna(0.0)
    for f in tqdm(dense_feats):
        df_dense[f] = df_dense[f].apply(lambda x: np.log(1+x) if x > -1 else -1)
        
    # sparse
    df_sparse = df[sparse_feats].fillna('-1')
    for f in tqdm(sparse_feats):
        lbe = LabelEncoder()  # 来自sklearn库，自动为类别特征打标签
        df_sparse[f] = lbe.fit_transform(df_sparse[f])

    df_new = pd.concat([df_dense, df_sparse], axis=1)
    return df_new

def main():
    path = '/home/zhuliqing/code/AI-RecommenderSystem/Recall/FM_FFM/FM/criteo/'
    data = pd.read_csv(path + 'train.csv')
    data.drop(['Id'], axis=1, inplace=True)
    cols = data.columns.values
    dense_feats = [f for f in cols if f[0] == 'I']
    sparse_feats = [f for f in cols if f[0] == 'C']

    # 数据预处理
    feats = process_feat(data, dense_feats, sparse_feats)
    
    # 划分训练和验证数据
    x_trn, x_tst, y_trn, y_tst = train_test_split(feats, data['Label'], test_size=0.2, random_state=2020)
    
    # 超参数
    input_dim = x_trn.shape[1]

    # 按批次读入数据
    bsz = 32
    epochs = 10
    train_len, test_len = x_trn.shape[0], x_tst.shape[0]
    iter_num = train_len // bsz

    with tf.Session() as sess:
        input_ph, labels_ph, loss = FM(input_dim)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss)
        sess.run(tf.global_variables_initializer())
        for epoch in tqdm(range(epochs)):
            for iter in range(iter_num + 1):
                # TODO:如果行数不够，就减少bsz或填充0
                feats = x_trn.iloc[iter * bsz: (iter + 1) * bsz].to_numpy()
                labels = y_trn.iloc[iter * bsz: (iter + 1) * bsz].to_numpy()
                labels = np.expand_dims(labels, axis=1).astype(np.float32)
                sess.run(train_op, feed_dict={input_ph: feats, labels_ph: labels})

                break
            break

if __name__ == '__main__':
    main()