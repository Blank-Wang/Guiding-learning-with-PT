import os
import configparser
from keras import backend as K
import keras
import tensorflow as tf
from keras.layers import Layer
from keras.models import load_model,Model
from keras.layers import Permute,Reshape,Lambda,Bidirectional,Conv2DTranspose,dot
from keras.layers import Embedding,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import Activation,BatchNormalization,TimeDistributed,Dropout
from keras.layers import GRU,Dense,Input,Activation,Conv2D,MaxPooling2D
from keras.layers import Dot,add,multiply,concatenate,subtract,GlobalMaxPooling1D
from keras.layers import UpSampling2D,GlobalMaxPooling2D
import numpy as np
et=0.0000001
 
class attentionLayer(Layer):
	def __init__(self,**kwargs):
		super(attentionLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(
				shape=(1,1,input_shape[2]),
				initializer=keras.initializers.Zeros(),
				name='%s_kernel'%self.name)
		self.bias = self.add_weight(
				shape=(1,1),
				initializer=keras.initializers.Zeros(),
				name='%s_bias'%self.name)

		super(attentionLayer, self).build(input_shape)
	
	def call(self, inputs):
		weights=K.sum(inputs*self.kernel,axis=-1)+self.bias
		return weights

	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[1])

class sumLayer(Layer):
	def __init__(self,axi,keep_dims,**kwargs):
		self.axi=axi
		self.keep_dims=keep_dims
		super(sumLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		super(sumLayer, self).build(input_shape)

	def call(self, inputs):
		axi=self.axi
		out=K.sum(inputs,axis=axi)
		out=K.expand_dims(out,axis=axi)
		return out

	def compute_output_shape(self, input_shape):
		axi=self.axi
		keep_dims=self.keep_dims
		if axi==1:
			if keep_dims:
				return (input_shape[0],1,input_shape[2])
			else:
				return (input_shape[0],input_shape[2])
		elif axi==2:
			if keep_dims:
				return (input_shape[0],input_shape[1],1)
			else:
				return (input_shape[0],input_shape[1])



class attend_cnn(object):

	def __init__(self,conf_dir,model_name,task_name):
		self.conf_dir=conf_dir
		self.model_name=model_name
		self.task_name=task_name
		self.init_model_conf()

	def set_DFs(self,dfs):
		self.dfs=dfs
	def set_CLASS(self,class_num):
		self.CLASS=class_num

	
	def init_model_conf(self):
		model_name=self.model_name
		conf_dir=self.conf_dir
		model_cfg_path=os.path.join(conf_dir,'model.cfg')

		assert os.path.exists(model_cfg_path)
		config=configparser.ConfigParser()
		config.read(model_cfg_path)
		assert model_name in config.sections()
		model_conf=config[model_name]

		self.model_mode=model_conf['model_mode']
		self.LEN=int(model_conf['LEN'])
		self.DIM=int(model_conf['DIM'])
		self.cnn_layer=int(model_conf['cnn_layer'])
		filters=model_conf['filters'].split(',')
		self.filters=[int(c) for c in filters]
		pool_size=model_conf['pool_size'].split(',')
		dilation_rate=model_conf['dilation_rate'].split(',')
		self.pool_size=[[int(p.split(' ')[0]),int(p.split(' ')[1])] 
					for p in pool_size]
		top_len=self.LEN
		for i in range(self.cnn_layer):
			top_len//=self.pool_size[i][0]
		self.top_len=top_len
		self.dilation_rate=[[int(p.split(' ')[0]),int(p.split(' ')[1])]
					for p in dilation_rate]
		conv_nums=model_conf['conv_nums'].split(',')
		self.conv_nums=[int(c) for c in conv_nums]
		kernel_size=model_conf['kernel_size'].split(',')
		self.kernel_size=[(int(k.split(' ')[0]),int(k.split(' ')[1])) for k in kernel_size]

			


		self.with_rnn=config.getboolean(model_name,'with_rnn')

		if self.with_rnn:
			self.hid_dim=int(model_conf['hid_dim'])

		self.with_dropout=config.getboolean(model_name,'with_dropout')
		if self.with_dropout:
			self.dr_rate=float(model_conf['dr_rate'])


	def cnn_block(self,dim,kernel_size,dr_mode,dr_rate,
			name,dilation_rate,pool_size,conv_num):
		result=[]
		cvs=[]
		md=self.model_name
		for i in range(conv_num):
			cvs+=[Conv2D(filters=dim,
					kernel_size=kernel_size,
					dilation_rate=dilation_rate,
					padding='same',
					name='%s_cnn_bl_conv_%d_%d'%(md,name,i)),
				BatchNormalization(axis=-1,name='%s_cnn_bl_bn_%d_%d'%(md,name,i)),
				Activation('relu',
					name='%s_cnn_bl_ac_%d_%d'%(md,name,i))]
			
		mp=MaxPooling2D(pool_size=pool_size,
			name='%s_cnn_bl_mp_%d'%(md,name))
		dr=Dropout(rate=dr_rate,name='%s_cnn_bl_dr_%d'%(md,name))

		result=cvs+[mp]
		if dr_mode:
			result+=[dr]
		return result


	def apply_layers(self,inputs,layers,cnt):
		out=inputs
		for layer in layers:
			out=layer(out)
		return out

	def graph(self):
		filters=self.filters
		cnn_layer=self.cnn_layer
		pool_size=self.pool_size
		dilation_rate=self.dilation_rate
		conv_nums=self.conv_nums
		kernel_size=self.kernel_size
		with_rnn=self.with_rnn
		with_dropout=self.with_dropout
		md=self.model_name

		CLASS=self.CLASS
		LEN=self.LEN
		DIM=self.DIM
		dfs=self.dfs
		top_len=LEN
		top_dim=DIM

		if with_dropout:
			dr_rate=self.dr_rate
		else:
			dr_rate=0

		cnns=[]
		for i in range(cnn_layer):
			cnns+=[self.cnn_block(filters[i],
					kernel_size[i],
					dr_mode=with_dropout,
					dr_rate=dr_rate,
					name=i,
					dilation_rate=dilation_rate[i],	
					pool_size=pool_size[i],
					conv_num=conv_nums[i])]

			top_len//=pool_size[i][0]
			top_dim//=pool_size[i][1]

		top_dim*=filters[-1]
		hidden=top_dim//CLASS
		h_dim=top_dim
		if with_rnn:
			hid_dim=self.hid_dim
			gru=GRU(hid_dim,name='%s_GRU'%md,
				return_sequences=True,dropout=0.1)
			bc=BatchNormalization(axis=-1,name='%s_gru_bc'%md)
			ac=Activation('relu',name='%s_gru_ac'%md)
			grus=[gru,bc,ac]
			h_dim=hid_dim
		denses=[]
		attens=[]
		hiddens=[]

		for i in range(CLASS):
			attens+=[attentionLayer(name='%s_atten_%d'%(md,i))]
			denses+=[Dense(1,use_bias=True,name='%s_Dense_%d'%(md,i))]
			hiddens+=[int(h_dim*dfs[i])]

		def create_model(inputs,test_mode=False):

			inputs=BatchNormalization(axis=-1,
				name='%s_BatchNormalization_input'%md)(inputs)
			out=Reshape([LEN,DIM,1])(inputs)
			outs=[]	
			for i in range(cnn_layer):
				out=self.apply_layers(out,cnns[i],len(cnns[i]))
		
			out=Reshape([top_len,top_dim])(out)	
			if with_rnn:
				out=self.apply_layers(out,grus,len(grus))

			outs=[]
			h_outs=[]	

			for i in range(CLASS):
				hidden=hiddens[i]
				tmp=Lambda(lambda x : x[:,:,:hidden])(out)

				ct_at=attens[i](tmp)

				ct_at=Reshape((top_len,1))(ct_at)
				h_out=Activation('sigmoid')(ct_at)

				ct_at=Reshape((top_len,))(ct_at)
				ct_at=Lambda(lambda x:x/hidden)(ct_at)
				ct_at=Activation('softmax')(ct_at)
				ct_at=Reshape((top_len,1))(ct_at)
				ct_at=multiply([ct_at,tmp])
				tmp=sumLayer(1,False)(ct_at)
				tmp=Reshape((hidden,))(tmp)
				tmp=denses[i](tmp)
				tmp=Activation('sigmoid')(tmp)
				h_out=Lambda(lambda x:x,name='%s_frame_output_%d'%(md,i))(h_out)
				h_outs+=[h_out]
				outs+=[tmp]
			out=concatenate(outs,axis=-1)
			h_out=concatenate(h_outs,axis=-1)
			if test_mode:
				return out,h_out
			return out
	
		return create_model


	def get_test_model(self,model,mode='sed'):
					
		model=self.get_model(pre_model=model,
					test_mode=True,mode=mode)
		return model
	
	def get_model(self,pre_model=None,test_mode=False,mode='at'):
		LEN=self.LEN
		DIM=self.DIM
		CLASS=self.CLASS
		inputs=Input((LEN,DIM))	
		creat_model=self.graph()
		if test_mode:
			out=creat_model(inputs,True)
			if (type(out) is tuple or type(outs) is list) and mode=='at':
				outs=out[0]
			else:
				outs=out
		else:	
			outs=creat_model(inputs,False)

		model=Model(inputs,outs)
		if pre_model is not None:
			if type(pre_model) is str:
				model.load_weights(pre_model,by_name=True)
			else:
				tmp_path='.%s-%s.h5'%(self.task_name,self.model_name)
				pre_model.save_weights(tmp_path)
				model.load_weights(tmp_path,by_name=True)
				os.remove(tmp_path)
		return model
