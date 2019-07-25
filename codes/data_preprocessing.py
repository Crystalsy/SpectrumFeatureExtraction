import numpy as np
import pandas as pd
import math
import random

def square(x):
    return x*x

lengths=np.asarray(pd.read_table("masses_len.txt",names=['length'],sep="\n")['length'].values.tolist())

m_x=[]
m_y=[]
for i in range(791364):
	m_x.append(np.asarray(list(map(float,raw_input().split()))))
for i in range(791364):
        m_y.append(np.asarray(list(map(float,raw_input().split()))))

index_list=np.arange(0,791364)
my_test=[]
sample_per_batch=np.zeros((56,))
length=[]

from collections import Counter
cnt=Counter(lengths)
listofpairs=pd.DataFrame(list(cnt.items()))
listofpairs.columns=['length','number']
listofpairs=listofpairs.sort_values(["length"],ascending=[True]).reset_index(drop=True)

for i in range(56):
	ith_length=listofpairs.iloc[i]['length']
	m_index=np.where(lengths==ith_length)[0]
	length.append(ith_length)
	sample_per_batch[i]=len(m_index)
	matched=index_list[np.isin(index_list,m_index)]
	num=len(m_index)
	temp_x=np.zeros((num,ith_length,1))
	temp_y=np.zeros((num,ith_length,1))	
	for k in range(num):
	    index=matched[k]
	    sorted_index=sorted(range(len(m_x[index])), key=lambda k: m_x[index][k])
	    for j in range(len(sorted_index)):#same as ith_length
	    	temp_x[k][j]=m_x[index][sorted_index[j]]
		temp_y[k][j]=m_y[index][sorted_index[j]]
	temp=np.concatenate((temp_x,temp_y),axis=2)
	my_test.append(temp)		

def randindex(df,n_matrix,test_size):
	indexlist=[]
	for i in range(len(df)):
		if(n_matrix[i]>=10):
			ntrn = round(n_matrix[i] * (test_size))
                        ntrn = int(ntrn)
                        index=np.int32(np.arange(0,n_matrix[i]))
                        random.shuffle(index)
			indexlist.append(index)
		else:
			indexlist.append(np.zeros((1,)))
	return indexlist

#divide test,train,val to 1,9,1
def train_test_split(df, test_size, n_matrix, indexlists):
        X_train=[]
        X_test=[]
        X_val=[]
	only_train=[]
        len_num=np.zeros((len(df)))
	test_num=np.zeros((len(df)))
        for i in range(len(df)):#775 or 56
		if(n_matrix[i]>=10):
			ntrn = round(n_matrix[i] * (test_size))
                        ntrn = int(ntrn)
			index=indexlists[i]
	                testindex=index[0:ntrn]
        	        valindex=index[ntrn:ntrn+ntrn]
                	trainindex=index[ntrn+ntrn:]
			test_d=np.zeros((ntrn,df[i].shape[1],2))
			val_d=np.zeros((ntrn,df[i].shape[1],2))
			train_d=np.zeros((int(n_matrix[i]-(ntrn*2)),int(df[i].shape[1]),2))
			for j in range(ntrn):
				test_d[j]=df[i][testindex[j]]
				val_d[j]=df[i][valindex[j]]
			for j in range(int(n_matrix[i]-(ntrn*2))):
				train_d[j]=df[i][trainindex[j]]
			X_test.append(test_d)
			X_val.append(val_d)
			X_train.append(train_d)
                	len_num[i]=n_matrix[i]-(ntrn+ntrn)
			test_num[i]=ntrn
		elif((n_matrix[i]>=3)&(n_matrix[i]<10)):
			X_test.append(df[i][0:1])
			X_val.append(df[i][1:2])
                        X_train.append(df[i][2:])
                        len_num[i]=n_matrix[i]-2
			test_num[i]=1
		else:
			X_train.append(df[i][0:])
			only_train.append(i)
			len_num[i]=n_matrix[i]
			X_val.append(np.zeros((2,)))
			X_test.append(np.zeros((2,)))
        return (X_train, X_test, X_val, len_num, test_num, only_train)

index_lists=randindex(my_test,np.int32(sample_per_batch),0.1)
length_of_sequences = sum(sample_per_batch)
(X_train, X_test, val, len_num, test_num, only_train) = train_test_split(my_test,test_size=0.1, n_matrix=np.int32(sample_per_batch),indexlists=index_lists)

def minmax(mylist,axis):
	if(axis==0):
        	min_val=mylist[0][:,:,0].min()
        	max_val=mylist[0][:,:,0].max()
        	for i in range(len(mylist)):
                	min_val=min(min_val,mylist[i][:,:,0].min())
                	max_val=max(max_val,mylist[i][:,:,0].max())
	else:
		min_val=mylist[0][:,:,1].min()
                max_val=mylist[0][:,:,1].max()
                for i in range(len(mylist)):
                        min_val=min(min_val,mylist[i][:,:,1].min())
                        max_val=max(max_val,mylist[i][:,:,1].max())
        return (min_val, max_val)

x_min_val, x_max_val=minmax(X_train,0)
y_min_val, y_max_val=minmax(X_train,1)

import copy
z_input=copy.deepcopy(my_test)
for i in range(len(my_test)):
	z_input[i][:,:,0]=((my_test[i][:,:,0]-x_min_val)/(x_max_val-x_min_val))
	z_input[i][:,:,1]=((my_test[i][:,:,1]-y_min_val)/(y_max_val-y_min_val))

(X_train,X_test,val,len_num, test_num, only_train)=train_test_split(z_input,test_size=0.1, n_matrix=sample_per_batch,indexlists=index_lists)

f1=open("bixy_traindata.txt",'w')
f2=open("bixy_testdata.txt",'w')
f3=open("bixy_val.txt",'w')
f4=open("bixy_len_num.txt",'w')
f5=open("bixy_test_num.txt",'w')
f6=open("bixy_train_only.txt",'w')


for i in range(56):
	if(np.isin(i,only_train)==False):
		for j in range(int(round(len_num[i]))):
			for k in range(length[i]):
				data1="%f "%X_train[i][j][k][0]
				data2="%f\n"%X_train[i][j][k][1]
				f1.write(data1+data2)
		for j in range(int(round(test_num[i]))):
			for k in range(length[i]):
                        	data1="%f "%X_test[i][j][k][0]
				data2="%f\n"%X_test[i][j][k][1]
                        	f2.write(data1+data2)
		for j in range(int(round(test_num[i]))):
                	for k in range(length[i]):
                        	data1="%f "%val[i][j][k][0]
				data2="%f\n"%val[i][j][k][1]
                        	f3.write(data1+data2)
	else:
		for j in range(int(round(len_num[i]))):
                	for k in range(length[i]):
                        	data1="%f "%X_train[i][j][k][0]
				data2="%f\n"%X_train[i][j][k][1]
                        	f1.write(data1+data2)
        	data="0 0"+str("\n")
        	f2.write(data)
		f3.write(data)
		f6.write(str(i)+str("\n"))
	data="%d\n"%len_num[i]
	f4.write(data)
	data="%d\n"%test_num[i]
	f5.write(data)


f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
