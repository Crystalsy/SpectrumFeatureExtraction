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
f1=open("pad_mytied_bilstm_autopredtrain.txt",'w')
f2=open("pad_mytied_bilstm_autopredtest.txt",'w')
f3=open("pad_mytied_bilstm_autopredval.txt",'w')

# In[9]:

with tf.device("/device:GPU:1"):
		start_time=time.time()
		timesteps=62
		inputs = Input(shape=(timesteps, input_dim))
                before_encoder=Dense(latent_dim,kernel_initializer=initializers.glorot_normal(seed=1))(inputs)
                #encoder
                for k in range(1):
                        encoded_lstm = Bidirectional(LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", kernel_initializer=keras.initializers.glorot_normal(seed=0),recurrent_initializer=initializers.Orthogonal(gain=1.0,seed=0),return_sequences=False))
                        encoded=encoded_lstm(before_encoder)
                #decoder
                hidden=RepeatVector(timesteps)(encoded)
                decoded=Dense(input_dim, kernel_initializer=initializers.glorot_normal(seed=2),activation="sigmoid")(hidden)

                BILSTM=Model(inputs,decoded)
                BILSTM.compile(optimizer='rmsprop', loss='mse',  metrics=['mae','mape'])
		batch=int(round(633082/100))
		a=BILSTM.fit(x=X_train,y=X_train, epochs=30, batch_size=batch, verbose=2, initial_epoch=0, shuffle=True, validation_data=(val,val))
		BILSTM.save_weights("pad_noreverse_bilstm_autoencodermodel.h5")
		print(BILSTM.summary())
		print(time.time() - start_time)
		test_batch=int(round(79141/10))
		#revised temp: just to check. X_*_rev to X_*
		X_hat_test=BILSTM.predict(X_test,batch_size=test_batch)
		X_train_pred=BILSTM.predict(X_train,batch_size=batch)
		val_pred=BILSTM.predict(val,batch_size=test_batch)

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

