#tested until 2019.02.17-changed customized.py to match more datasets
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
from tensorflow.keras import initializers
from keras.models import load_model, model_from_json
from keras import backend as K
import keras

input_dim=1
latent_dim=100
init=1
#0 random 1 encoder

with tf.device("/device:GPU:1"):
    train_result=[]
    val_result=[]
    X_test_rev=[]
    X_hat_test=[]
    for j in range(30):
	for i in range(batch_size):
	    K.clear_session()
	    with tf.device("/device:GPU:1"):
		timesteps=int(listofpairs.iloc[index[i]]['length'])
        	inputs = Input(shape=(timesteps, input_dim))
		if(init==0):
			#random init
			before_encoder=Dense(latent_dim,kernel_initializer=initializers.glorot_normal(seed=1))(inputs)
			#encoder
			for k in range(1):
				encoded_lstm = Bidirectional(LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),
                recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_sequences=False))
				encoded=encoded_lstm(before_encoder)
        		#decoder
			hidden=RepeatVector(timesteps)(encoded)
			for k in range(1):
				decoded_lstm_1= LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),
                recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True)
				decoded_1=decoded_lstm_1(hidden)
				decoded_lstm_2=LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),
                recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True,go_backwards=True)
				decoded_2=decoded_lstm_2(hidden)
				merged_decoder=concatenate([decoded_1,decoded_2])
			after_decoder=Dense(input_dim, kernel_initializer=initializers.glorot_normal(seed=2),activation="sigmoid")(merged_decoder)
                elif(init==1):
                        #encoder init
			before_encoder=Dense(latent_dim,kernel_initializer=initializers.glorot_normal(seed=1))(inputs)
                        #encoder
                        for k in range(1):
                                encoded_lstm = Bidirectional(LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),
                recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_sequences=False,return_state=True))
                                encoded,forward_h, forward_c, backward_h, backward_c=encoded_lstm(before_encoder)
                                forward_encoder_state=[forward_h,forward_c]
                                backward_encoder_state=[backward_h,backward_c]

                        #decoder
                        hidden=RepeatVector(timesteps)(encoded)
                        for k in range(1):
                                decoded_lstm_1= LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),
                recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True,return_state=True)
                                decoded_1,de_h_1,de_c_1=decoded_lstm_1(hidden,initial_state=forward_encoder_state)
                                decoded_lstm_2=LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),
                recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True,return_state=True,go_backwards=True)
                                decoded_2,de_h_2,de_c_2=decoded_lstm_2(hidden,initial_state=backward_encoder_state)
                                merged_decoder=concatenate([decoded_1,decoded_2])
			after_decoder=Dense(input_dim, kernel_initializer=initializers.glorot_normal(seed=2),activation="sigmoid")(merged_decoder)

		#first_dense=Dense(latent_dim, kernel_initializer=initializers.glorot_normal(seed=0),activation="relu")(merged_decoder)
                #output=Dense(input_dim, kernel_initializer=initializers.glorot_normal(seed=0),activation="sigmoid")(first_dense)
        	#output=Dense(input_dim,kernel_initializer=initializers.glorot_normal(seed=0),activation="sigmoid")(merged_decoder)
		BILSTM_AE=Model(inputs,after_decoder)
		BILSTM_AE.compile(optimizer='rmsprop', loss='mse',  metrics=['mae','mape'])
		if((i>0)|(j>0)):
                        BILSTM_AE.load_weights("previous_weights.h5")
                if(np.isin(index[i],only_train)==False):#val also exist
                        if(len_num[index[i]]<10):
                                batch=int(round(len_num[index[i]]))
                        elif((len_num[index[i]]>=10)&(len_num[index[i]]<100)):
                                batch=int(round(len_num[index[i]]/10))
                        else:
                                batch=int(round(len_num[index[i]]/100))
                        a=BILSTM_AE.fit(x=X_train[i], y=X_train[i], epochs=1, batch_size=batch, shuffle=True, validation_data=(val[i],val[i]))
                	BILSTM_AE.save_weights("previous_weights.h5")
		else:
                        batch=int(round(len_num[index[i]]))
                        a=BILSTM_AE.fit(x=X_train[i],y=X_train[i], epochs=1, batch_size=batch, shuffle=True)
                	BILSTM_AE.save_weights("previous_weights.h5")
		print(BILSTM_AE.summary())
	model_name="mytied_encoder_biauto_ver1_hidden100_"+str(j)+".h5"
        BILSTM_AE.save_weights(model_name)
        print("*******\n")
        print(time.time() - start_time)
