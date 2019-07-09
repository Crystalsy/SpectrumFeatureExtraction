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

train=pd.read_table('pad_traindata.txt',names=['x'])['x'].values.tolist()
test=pd.read_table('pad_testdata.txt',names=['x'])['x'].values.tolist()
valid=pd.read_table('pad_val.txt',names=['x'])['x'].values.tolist()
len_num=np.int32(pd.read_table('pad_len_num.txt',names=['x'])['x'].values.tolist())
test_num=np.int32(pd.read_table('pad_test_num.txt',names=['x'])['x'].values.tolist())
only_train=pd.read_table('pad_train_only.txt',names=['x'])['x'].values.tolist()

X_train=np.zeros((int(round(np.sum(len_num))),62,1))
val=np.zeros((int(round(np.sum(test_num))),62,1))
X_test=np.zeros((int(round(np.sum(test_num))),62,1))

tr=0
te=0
va=0

batch_size=56

for i in range(int(round(np.sum(len_num)))):
                X_train[i]=np.asarray(train[tr:tr+62]).reshape(62,1)
                tr=tr+62

for i in range(int(round(np.sum(test_num)))):
                X_test[i]=np.asarray(test[te:te+62]).reshape(62,1)
                val[i]=np.asarray(valid[va:va+62]).reshape(62,1)
                te=te+62
                va=va+62
#already randomized data

from keras.layers import Input, LSTM, RepeatVector, concatenate, Dense,Bidirectional,Masking, Activation,TimeDistributed
from keras.models import Model,Sequential
from keras.callbacks import LambdaCallback
from tensorflow.keras import initializers
#from tensorflow.keras.models import load_model, model_from_json,Sequential
from keras import backend as K
import keras

input_dim=1
latent_dim=100
timesteps=62
init=1
f1=open("pad_mytied_biauto_autopredtrain.txt",'w')
f2=open("pad_mytied_biauto_autopredtest.txt",'w')
f3=open("pad_mytied_biauto_autopredval.txt",'w')

# In[9]:

with tf.device("/device:GPU:1"):
		start_time=time.time()
		timesteps=62
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
		BILSTM_AE=Model(inputs,after_decoder)
                BILSTM_AE.compile(optimizer='rmsprop', loss='mse',  metrics=['mae','mape'])
		batch=int(round(633082/100))
		a=BILSTM_AE.fit(x=X_train,y=X_train, epochs=30, batch_size=batch, verbose=2, initial_epoch=0, shuffle=True, validation_data=(val,val))
		BILSTM_AE.save_weights("pad_mytied_noreverse_biauto_autoencodermodel.h5")
		print(BILSTM_AE.summary())
		print(time.time() - start_time)
		test_batch=int(round(79141/10))
		#revised temp: just to check. X_*_rev to X_*
		X_hat_test=BILSTM_AE.predict(X_test,batch_size=test_batch)
		X_train_pred=BILSTM_AE.predict(X_train,batch_size=batch)
		val_pred=BILSTM_AE.predict(val,batch_size=test_batch)

for i in range(int(round(np.sum(len_num)))):
	for j in range(62):
		data="%f\n"%X_train_pred[i][j]
		f1.write(data)
for i in range(int(round(np.sum(test_num)))):
	for j in range(62):
		data="%f\n"%X_hat_test[i][j]
		f2.write(data)
		data="%f\n"%val_pred[i][j]
		f3.write(data)
		
f1.close()
f2.close()
f3.close()

