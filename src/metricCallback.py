import configparser
from keras import backend as K
import keras
from keras import objectives
import tensorflow as tf
from keras.models import load_model,Model
import os
import numpy as np
from src import utils
from src.Logger import LOG

class metricCallback(keras.callbacks.Callback):
	def __init__(self,conf_dir,train_mode='semi'):
		""""
		MetricCallback for training.
		Args:
		Attributes:
			conf_dir
			train_mode
			learning_rate
			decay_rate
			epoch_of_decay
			early_stop
			metric
			ave
			f1_utils
			best_model_path
			batch_size
			CLASS
			best_f1
			best_epoch
			wait
		Interface:
			set_extra_attributes
			init_attributes
			check_attributes
			init_train_conf
			get_at
			get_opt
			get_loss	
			on_train_begin
			on_epoch_end
			on_train_end

                """	
		self.train_mode=train_mode
		self.conf_dir=conf_dir
		self.init_train_conf()
		self.init_attributes()
		super(metricCallback, self).__init__()

		
	def set_extra_attributes(self,f1_utils,best_model_path,batch_size,CLASS):
		self.f1_utils=f1_utils
		self.best_model_path=best_model_path
		self.batch_size=batch_size
		self.CLASS=CLASS

	def init_attributes(self):
		self.best_f1=-1
		self.best_epoch=-1
		self.wait=0

	def check_attributes(self):
		attributes=[self.f1_utils,
			self.best_model_path,
			self.batch_size,
			self.CLASS]

		for attribute in attributes:
			assert attribute is not None

	def init_train_conf(self):
		conf_dir=self.conf_dir
		train_cfg_path=os.path.join(conf_dir,'train.cfg')
		assert os.path.exists(train_cfg_path)
		config=configparser.ConfigParser()
		config.read(train_cfg_path)

		assert 'metricCallback' in config.sections()
		train_conf=config['metricCallback']
		self.learning_rate=float(train_conf['learning_rate'])
		self.decay_rate=float(train_conf['decay_rate'])
		self.epoch_of_decay=int(train_conf['epoch_of_decay'])
		self.early_stop=int(train_conf['early_stop'])
		assert 'validate' in config.sections()
		vali_conf=config['validate']
		self.metric=vali_conf['metric']
		self.ave=vali_conf['ave']

	def get_at(self,preds,labels):
		f1_utils=self.f1_utils
		f1,_,_=f1_utils.get_f1(preds,labels,mode='at')
		return f1

	def get_opt(self,lr):
		opt=keras.optimizers.Adam(lr=lr,beta_1=0.9,
			beta_2=0.999,epsilon=1e-8,decay=1e-8)
		return opt

	def get_loss(self):
		CLASS=self.CLASS
		train_mode=self.train_mode

		def loss(y_true,y_pred):
			return K.mean(K.binary_crossentropy(y_true[:,:CLASS],
						y_pred),axis=-1)

		def semi_loss(y_true,y_pred):

			a=y_true[:,CLASS*2:CLASS*2+1]
			mask=y_true[:,CLASS:CLASS*2]
			y_true=y_true[:,:CLASS]
			y_pred_1=y_pred[:,:CLASS]
			y_pred_2=y_pred[:,CLASS:]

			y_pred_2_X=K.relu(K.relu(y_pred_2,threshold=0.5)*2,
					max_value=1)
			y_pred_1_X=K.relu(K.relu(y_pred_1,threshold=0.5)*2,
					max_value=1)

			closs=K.mean(K.binary_crossentropy(
					y_true*mask,y_pred_1*mask),axis=-1)
			closs+=K.mean(K.binary_crossentropy(
					y_true*mask,y_pred_2*mask),axis=-1)
			mask=1-mask

			closs+=K.mean(K.binary_crossentropy(
				y_pred_1_X*mask,y_pred_2*mask),axis=-1)
			closs+=K.mean(K.binary_crossentropy(
				y_pred_2_X*mask*a,y_pred_1*mask*a),axis=-1)
			return closs

		if train_mode=='supervised':
			return loss
		elif train_mode=='semi':
			return semi_loss
		assert True

	def on_train_begin(self, logs={}):
		self.check_attributes()
		LOG.info('init training...')
		LOG.info('metrics : %s %s'%(self.metric,self.ave))

		opt=self.get_opt(self.learning_rate)
		loss=self.get_loss()
		self.model.compile(optimizer=opt,loss=loss)
	
	def on_epoch_end(self, epoch, logs={}):

		best_f1=self.best_f1
		f1_utils=self.f1_utils
		CLASS=self.CLASS
		train_mode=self.train_mode
		early_stop=self.early_stop

		vali_data=self.validation_data
		labels=vali_data[1][:,:CLASS]

		preds=self.model.predict(vali_data[0],batch_size=self.batch_size)
		
		if train_mode=='semi':
			preds_PT=preds[:,:CLASS]
			preds_PS=preds[:,CLASS:]
			pt_f1=self.get_at(preds_PT,labels)
			ps_f1=self.get_at(preds_PS,labels)
		else:
			ps_f1=self.get_at(preds,labels)

		logs['f1_val']=ps_f1

		is_best='not_best'		
		if logs['f1_val']>=self.best_f1:
			self.best_f1=logs['f1_val']
			self.best_epoch=epoch
			self.model.save_weights(self.best_model_path)
			is_best='best'
			self.wait=0

		self.wait+=1
		if self.wait>early_stop:
			self.stopped_epoch = epoch
			self.model.stop_training = True

		if train_mode=='semi':
			LOG.info('[ epoch %d ,sed f1 : %f , at f1 : %f ] %s'
				%(epoch,logs['f1_val'],pt_f1,is_best))
		else:
			LOG.info('[ epoch %d ,%s f1 : %f ] %s'
				%(epoch,mode,logs['f1_val'],is_best))

		if epoch>0 and epoch%self.epoch_of_decay==0:
			self.learning_rate*=self.decay_rate
			opt=self.get_opt(self.learning_rate)
			LOG.info('[ epoch %d ,learning rate decay to %f ]'%(
					epoch,learning_rate))
			loss=self.get_loss()
			self.model.compile(optimizer=opt,loss=loss)
		
		
	def on_train_end(self, logs={}):
		best_epoch=self.best_epoch
		best_f1=self.best_f1
		LOG.info('[ best vali f1 : %f at epoch %d ]'%(best_f1,best_epoch))

	
