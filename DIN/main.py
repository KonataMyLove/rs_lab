from DIN import DIN
from data_crecate import create_amazon_electronic_dataset

file_name = '/home/zhuliqing/code/AI-RecommenderSystem/Rank/DIN/TraditionalTFStyle/dataset/remap.pkl'
feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y) = create_amazon_electronic_dataset(file_name)

att_hidden_units = [80, 40]
ffn_hidden_units = [256, 128, 64]
att_activation = 'sigmoid'
maxlen = 40
dnn_dropout = 0.5
model = DIN(feature_columns, behavior_list, att_hidden_units=att_hidden_units,  
            ffn_hidden_units=ffn_hidden_units, maxlen=maxlen, dropout=dnn_dropout)
model.summary()