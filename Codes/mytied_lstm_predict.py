#tested until 2019.06.15-Not encoder decoder network!!
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import time
import random

start_time=time.time()
def square(x):
    return x*x

lengths=np.asarray(pd.read_table("masses_len.txt",names=['length'],sep="\n")['length'].values.tolist())


from collections import Counter
cnt=Counter(lengths)
listofpairs=pd.DataFrame(list(cnt.items()))
listofpairs.columns=['length','number']
listofpairs=listofpairs.sort_values(["length"],ascending=[True]).reset_index(drop=True)

train=pd.read_table('bi_traindata.txt',names=['x'])['x'].values.tolist()
test=pd.read_table('bi_testdata.txt',names=['x'])['x'].values.tolist()
valid=pd.read_table('bi_val.txt',names=['x'])['x'].values.tolist()
len_num=np.int32(pd.read_table('bi_len_num.txt',names=['x'])['x'].values.tolist())
test_num=np.int32(pd.read_table('bi_test_num.txt',names=['x'])['x'].values.tolist())
only_train=pd.read_table('bi_train_only.txt',names=['x'])['x'].values.tolist()

temp_X_train=[]
temp_X_test=[]
temp_val=[]

tr=0
te=0
va=0

batch_size=56


for i in range(batch_size):
        if(np.isin(i,only_train)==False):
                a=int(len_num[i])*int(listofpairs.iloc[i]['length'])
                temp_X_train.append(np.asarray(train[tr:tr+a]).reshape(int(len_num[i]),int(listofpairs.iloc[i]['length']),1))
                tr=tr+a
                b=int(test_num[i])*int(listofpairs.iloc[i]['length'])
                temp_X_test.append(np.asarray(test[te:te+b]).reshape(int(test_num[i]),int(listofpairs.iloc[i]['length']),1))
                temp_val.append(np.asarray(valid[va:va+b]).reshape(int(test_num[i]),int(listofpairs.iloc[i]['length']),1))
                te=te+b
                va=va+b
        else:
                a=int(len_num[i])*int(listofpairs.iloc[i]['length'])
                temp_X_train.append(np.asarray(train[tr:tr+a]).reshape(int(len_num[i]),int(listofpairs.iloc[i]['length']),1))
                tr=tr+a
                b=int(test_num[i])*int(listofpairs.iloc[i]['length'])+1
                temp_X_test.append(np.asarray(test[te:te+b]))
                temp_val.append(np.asarray(valid[va:va+b]))
                te=te+b
                va=va+b

index=list(map(int,raw_input().split()))

#index=range(0,56)

X_train=[]
X_test=[]
val=[]
for i in range(batch_size):
	this=index[i]
	X_train.append(temp_X_train[this])
	X_test.append(temp_X_test[this])
	val.append(temp_val[this])


from keras.layers import Input, LSTM, RepeatVector, concatenate, Dense, Masking, Activation, Bidirectional, TimeDistributed
from keras.models import Model, Sequential
from keras.callbacks import LambdaCallback
from tensorflow.python.keras import initializers
from keras.models import model_from_json
from keras import backend as K
import keras

input_dim=1
latent_dim=100

with tf.device("/device:GPU:1"):
    train_result=[]
    val_result=[]
    X_test_rev=[]
    X_hat_test=[]
    for j in range(30):
	model_name="mytied_encoder_lstmonly_hidden100_"+str(j)+".h5"
	for i in range(batch_size):
	    K.clear_session()
	    with tf.device("/device:GPU:1"):
		timesteps=int(listofpairs.iloc[index[i]]['length'])
		# encode
                inputs = Input(shape=(timesteps, input_dim))
		before_encoder=Dense(latent_dim,kernel_initializer=initializers.glorot_normal(seed=1))(inputs)
                encoded= LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=initializers.glorot_normal(seed=0),
                recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_sequences=False)(before_encoder)

                #decode
                hidden = RepeatVector(timesteps)(encoded)
                decoded = Dense(input_dim,kernel_initializer=initializers.glorot_normal(seed=2), activation="sigmoid")(hidden)
		LSTM_ONLY=Model(inputs,decoded)
		LSTM_ONLY.load_weights(model_name)
		LSTM_ONLY.compile(optimizer='rmsprop',loss='mse',metrics=['mae','mape'])
		if(np.isin(index[i],only_train)==False):#val also exist
                        train_result.append(LSTM_ONLY.predict(X_train[i],batch_size=int(round(len_num[index[i]]))))
                        val_result.append(LSTM_ONLY.predict(val[i],batch_size=int(round(test_num[index[i]]))))
                        X_hat_test.append(LSTM_ONLY.predict(X_test[i],batch_size=int(round(test_num[index[i]]))))
                else:
                        batch=int(round(len_num[index[i]]))
                        train_result.append(LSTM_ONLY.predict(X_train[i],batch_size=batch))
                        val_result.append(np.zeros((1,)))
                        X_hat_test.append(np.zeros((1,)))
        print("____________\n")
        print(time.time() - start_time)

f1=open("mytied_encoder_lstmonly_han_unit100_batch100_pred_train.txt",'w')
f2=open("mytied_encoder_lstmonly_han_unit100_batch100_pred_val.txt",'w')
f3=open("mytied_encoder_lstmonly_han_unit100_batch100_pred_test.txt",'w')

for i in range(30):
        for j in range(56):
                if(np.isin(index[j], only_train)==False):
                        for k in range(int(round(len_num[index[j]]))):
                                for l in range(int(listofpairs.iloc[index[j]]['length'])):
                                        data="%f\n"%train_result[56*i+j][k][l][0]
                                        f1.write(data)
                        for k in range(int(round(test_num[index[j]]))):
                                for l in range(int(listofpairs.iloc[index[j]]['length'])):
                                        data="%f\n"%val_result[56*i+j][k][l][0]
                                        f2.write(data)
                                        data="%f\n"%X_hat_test[56*i+j][k][l][0]
                                        f3.write(data)
                else:
                        for k in range(int(round(len_num[index[j]]))):
                                for l in range(int(listofpairs.iloc[index[j]]['length'])):
                                        data="%f\n"%train_result[56*i+j][k][l][0]
                                        f1.write(data)

                        f2.write(str(0)+str("\n"))
                        f3.write(str(0)+str("\n"))

f1.close()
f2.close()
f3.close()
