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

train=pd.read_table('bixy_traindata.txt',names=['x','y'],sep=" ")
test=pd.read_table('bixy_testdata.txt',names=['x','y'],sep=" ")
valid=pd.read_table('bixy_val.txt',names=['x','y'],sep=" ")
len_num=np.int32(pd.read_table('bixy_len_num.txt',names=['x'])['x'].values.tolist())
test_num=np.int32(pd.read_table('bixy_test_num.txt',names=['x'])['x'].values.tolist())
only_train=pd.read_table('bixy_train_only.txt',names=['x'])['x'].values.tolist()

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
                temp_X_train.append(np.asarray(train.iloc[tr:tr+a]).reshape(int(len_num[i]),int(listofpairs.iloc[i]['length']),2))
                tr=tr+a
                b=int(test_num[i])*int(listofpairs.iloc[i]['length'])
                temp_X_test.append(np.asarray(test.iloc[te:te+b]).reshape(int(test_num[i]),int(listofpairs.iloc[i]['length']),2))
                temp_val.append(np.asarray(valid.iloc[va:va+b]).reshape(int(test_num[i]),int(listofpairs.iloc[i]['length']),2))
                te=te+b
                va=va+b
        else:
                a=int(len_num[i])*int(listofpairs.iloc[i]['length'])
                temp_X_train.append(np.asarray(train.iloc[tr:tr+a]).reshape(int(len_num[i]),int(listofpairs.iloc[i]['length']),2))
                tr=tr+a
                b=int(test_num[i])*int(listofpairs.iloc[i]['length'])+1
                temp_X_test.append(np.asarray(test.iloc[te:te+b]).reshape(2,))
                temp_val.append(np.asarray(valid.iloc[va:va+b]).reshape(2,))
                te=te+b
                va=va+b


index=list(map(int,raw_input().split()))

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

input_dim=2
latent_dim=200
init=1
#0 random 1 encoder

with tf.device("/device:GPU:0"):
    train_result=[]
    val_result=[]
    X_test_rev=[]
    X_hat_test=[]
    for j in range(100):
	for i in range(batch_size):
	    K.clear_session()
	    with tf.device("/device:GPU:0"):
		timesteps=int(listofpairs.iloc[index[i]]['length'])
        	inputs = Input(shape=(timesteps, input_dim))
		if(init==0):
			#random init
			before_encoder=Dense(latent_dim,kernel_initializer=initializers.glorot_normal(seed=1))(inputs)
			#encoder
			encoded_lstm_1 = Bidirectional(LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_sequences=True))
			encoded_1=encoded_lstm_1(before_encoder)
			encoded_lstm_2 = Bidirectional(LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_sequences=False))
                        encoded_2=encoded_lstm_2(encoded_1)

        		#decoder
			hidden=RepeatVector(timesteps)(encoded_2)
			first_decoded_lstm_1= LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True)
			first_decoded_1=first_decoded_lstm_1(hidden)
			first_decoded_lstm_2=LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True,go_backwards=True)
			first_decoded_2=first_decoded_lstm_2(hidden)
			merged_decoder=concatenate([first_decoded_1,first_decoded_2])
			
			second_decoded_lstm_1= LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True)
                        second_decoded_1=second_decoded_lstm_1(merged_decoder)
                        second_decoded_lstm_2=LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True,go_backwards=True)
                        second_decoded_2=second_decoded_lstm_2(merged_decoder)
                        final_decoder=concatenate([second_decoded_1,second_decoded_2])
                        after_decoder=Dense(input_dim, kernel_initializer=initializers.glorot_normal(seed=2),activation="sigmoid")(final_decoder)

                elif(init==1):
                        #encoder init
			before_encoder=Dense(latent_dim,kernel_initializer=initializers.glorot_normal(seed=1))(inputs)
                        #encoder
                        encoded_lstm_1 = Bidirectional(LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_state=True,return_sequences=True))
                        encoded_1,forward_h_1, forward_c_1, backward_h_1, backward_c_1=encoded_lstm_1(before_encoder)
                        forward_encoder_state_1=[forward_h_1,forward_c_1]
                        backward_encoder_state_1=[backward_h_1,backward_c_1]

			encoded_lstm_2 = Bidirectional(LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_state=True,return_sequences=False))
                        encoded_2,forward_h_2, forward_c_2, backward_h_2, backward_c_2=encoded_lstm_2(encoded_1)
			forward_encoder_state_2=[forward_h_2,forward_c_2]
                        backward_encoder_state_2=[backward_h_2,backward_c_2]


                        #decoder
                        hidden=RepeatVector(timesteps)(encoded_2)
                        first_decoded_lstm_1= LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True)
                        first_decoded_1=first_decoded_lstm_1(hidden,initial_state=forward_encoder_state_1)
                        first_decoded_lstm_2=LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True,go_backwards=True)
                        first_decoded_2=first_decoded_lstm_2(hidden,initial_state=backward_encoder_state_1)
                        merged_decoder=concatenate([first_decoded_1,first_decoded_2])

                        second_decoded_lstm_1= LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True)
                        second_decoded_1=second_decoded_lstm_1(merged_decoder,initial_state=forward_encoder_state_2)
                        second_decoded_lstm_2=LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),trainable=False,return_sequences=True,go_backwards=True)
                        second_decoded_2=second_decoded_lstm_2(merged_decoder,initial_state=backward_encoder_state_2)
                        final_decoder=concatenate([second_decoded_1,second_decoded_2])
                        after_decoder=Dense(input_dim, kernel_initializer=initializers.glorot_normal(seed=2),activation="sigmoid")(final_decoder)

		BILSTM_AE=Model(inputs,after_decoder)
		BILSTM_AE.compile(optimizer='rmsprop', loss='mse',  metrics=['mae','mape'])
		if((i>0)|(j>0)):
                        BILSTM_AE.load_weights("ver2_previous_weights.h5")
                if(np.isin(index[i],only_train)==False):#val also exist
                        if(len_num[index[i]]<10):
                                batch=int(round(len_num[index[i]]))
                        elif((len_num[index[i]]>=10)&(len_num[index[i]]<100)):
                                batch=int(round(len_num[index[i]]/10))
                        else:
                                batch=int(round(len_num[index[i]]/100))
                        a=BILSTM_AE.fit(x=X_train[i], y=X_train[i], epochs=1, batch_size=batch, shuffle=True, validation_data=(val[i],val[i]))
                	BILSTM_AE.save_weights("ver2_previous_weights.h5")
		else:
                        batch=int(round(len_num[index[i]]))
                        a=BILSTM_AE.fit(x=X_train[i],y=X_train[i], epochs=1, batch_size=batch, shuffle=True)
                	BILSTM_AE.save_weights("ver2_previous_weights.h5")
		print(BILSTM_AE.summary())
	model_name="xy_mytied_encoder_biauto_ver2_epoch100_hidden200_"+str(j)+".h5"
        BILSTM_AE.save_weights(model_name)
        print("*******\n")
        print(time.time() - start_time)
