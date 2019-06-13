import copy
import configparser
from keras import backend as K
import keras
from keras import objectives
import tensorflow as tf
from keras.models import load_model,Model
import os
import numpy as np
import random
import shutil
import sys
from src import data_loader as data
from src import model as md
from src import utils
from src import metricCallback
from src.Logger import LOG

class trainer(object):
	def __init__(self,task_name,model_name,from_exp):
		self.task_name=task_name
		self.model_name=model_name
		self.resume_training=from_exp
		if from_exp:
			self.conf_dir=os.path.join('exp',task_name,model_name,'conf')
		else:
			self.conf_dir=os.path.join(task_name,'conf')
		data_loader=self.init_data()
		model_struct=self.init_model(
			data_loader.LEN,data_loader.DIM,data_loader.CLASS)
		self.init_train_conf()
		self.prepare_exp()
		utils_obj=self.init_utils()

		dfs=data_loader.count_disentangle()
		model_struct.set_DFs(dfs)
		win_lens=data_loader.count_win_len_per_class(model_struct.top_len)
		utils_obj.set_win_lens(win_lens)
		data_loader.set_semi_parameter(self.exponent,self.start_epoch)
		data_loader.set_ep_per_epochs(self.ep_per_epochs)

	def init_train_conf(self):
		conf_dir=self.conf_dir
		train_cfg_path=os.path.join(conf_dir,'train.cfg')
		assert os.path.exists(train_cfg_path)
		config=configparser.ConfigParser()
		config.read(train_cfg_path)	
	
		assert 'trainer' in config.sections()
		train_conf=config['trainer']
		self.epochs=int(train_conf['epochs'])
		self.ep_per_epochs=float(train_conf['ep_per_epochs'])
		self.exponent=float(train_conf['exponent'])
		self.start_epoch=int(train_conf['start_epoch'])
		assert 'validate' in config.sections()
		vali_conf=config['validate']
		self.metric=vali_conf['metric']
		self.ave=vali_conf['ave']

	def init_model(self,LEN,DIM,CLASS):
		conf_dir=self.conf_dir
		model_name=self.model_name
		task_name=self.task_name
		self.model_struct=md.attend_cnn(conf_dir,model_name,task_name,
			LEN,DIM,CLASS)
		return self.model_struct

	def init_data(self):
		conf_dir=self.conf_dir
		self.data_loader=data.data_loader(conf_dir)
		return self.data_loader

	def init_utils(self):
		conf_dir=self.conf_dir
		data_loader=self.data_loader
		exp_dir=self.exp_dir
		self.utils=utils.utils(conf_dir,exp_dir,self.metric,
			self.ave,data_loader.events)
		lst,csv=data_loader.get_test()
		self.utils.init_csv(csv)
		lst,csv=data_loader.get_vali()
		self.utils.init_csv(csv)
		self.utils.set_vali_csv(lst,csv)
		return self.utils

	def init_metricCallback(self,train_mode):
		data_loader=self.data_loader
		callbacks=metricCallback.metricCallback(self.conf_dir,train_mode)
		callbacks.set_extra_attributes(self.utils,
						self.best_model_path,
						data_loader.batch_size,
						data_loader.CLASS)
		return callbacks

	def prepare_exp(self):
		model_name=self.model_name
		task_name=self.task_name
		resume_training=self.resume_training
		conf_dir=self.conf_dir
		if not os.path.exists('exp'):
			os.mkdir('exp')
		root_dir=os.path.join('exp',task_name)
		if not os.path.exists(root_dir):
			os.mkdir(root_dir)
		exp_dir=os.path.join(root_dir,model_name)
		model_dir=os.path.join(exp_dir,'model')
		result_dir=os.path.join(exp_dir,'result')
		exp_conf_dir=os.path.join(exp_dir,'conf')
		label_dir=os.path.join(exp_dir,'label')
		self.exp_dir=exp_dir
		self.result_dir=result_dir
		self.exp_conf_dir=exp_conf_dir
		self.best_model_path=os.path.join(model_dir,'best_model_w.h5')
		self.model_path=os.path.join(model_dir,'model_w.h5')
		self.label_dir=label_dir
		if not resume_training:
			if os.path.exists(exp_dir):
				shutil.rmtree(exp_dir)
			os.mkdir(exp_dir)
			os.mkdir(model_dir)
			os.mkdir(result_dir)
			os.mkdir(label_dir)
			shutil.copytree(conf_dir,exp_conf_dir)
			
		else:
			assert os.path.exists(exp_dir)
			assert os.path.exists(exp_conf_dir)
			assert os.path.exists(model_dir)
			if not os.path.exists(result_dir):
				os.mkdir(result_dir)
			if not os.path.exists(label_dir):
				os.mkdir(label_dir)



	def train(self,extra_model=None,train_mode='semi'):
		
		best_model_path=self.best_model_path
		epochs=self.epochs
		resume_training=self.resume_training
		model_name=self.model_name
		data_loader=self.data_loader

		callbacks=self.init_metricCallback(train_mode)

		if extra_model is not None:
			model=extra_model
		elif resume_training:
			model=model_struct.get_model(model_path)
		else:
			model=model_struct.get_model()

		model.compile(optimizer='Adam',loss='binary_crossentropy')

		gt,steps_per_epoch=self.data_loader.generator_train()
		vali_data=self.data_loader.generator_vali()

		model.fit_generator(gt(),steps_per_epoch=steps_per_epoch,
			epochs=epochs,shuffle=False,
			validation_data=vali_data,callbacks=[callbacks])


	def test_both(self):
		self.save_at_result()
		self.save_sed_result()


		
	

	def test(self,data_set,mode,preds={}):
		data_loader=self.data_loader
		assert data_set=='vali' or data_set=='test'
		if data_set=='vali':
			data=data_loader.generator_vali()
		else:
			data=data_loader.generator_test()

		assert mode=='at' or mode=='sed'
		best_model_path=self.best_model_path
		if mode=='at':
			if not data_set in preds:
		
				model=self.model_struct.get_model(
					pre_model=best_model_path,
					mode=mode)
				preds=model.predict(data[0],batch_size=data_loader.batch_size)
			else:
				preds=preds[data_set]
			return preds,data[1][:,:data_loader.CLASS]
		else:
			if not data_set in preds:
				model=self.model_struct.get_model(
					pre_model=best_model_path,
					mode=mode)
			#model.summary()
				preds=model.predict(data[0],batch_size=data_loader.batch_size)
			else:
				preds=preds[data_set]
			return preds[0],preds[1]
		
	
	def save_at_result(self,at_preds={}):
		preds_out={}
		preds_out['vali']=self.save_at('vali',at_preds,is_add=False)
		preds_out['test']=self.save_at('test',at_preds,is_add=True)
		return preds_out
		

	def save_at(self,mode='test',at_preds={},is_add=False):
		result_dir=self.result_dir
		model_name=self.model_name
		data_loader=self.data_loader
		f1_utils=self.utils
		result_path=os.path.join(result_dir,model_name+'_at.txt')

		if mode=='vali':
			lst,csv=data_loader.get_vali()
		elif mode=='test':
			lst,csv=data_loader.get_test()
		f1_utils.set_vali_csv(lst,csv)

		preds,labels=self.test(mode,'at',at_preds)
		preds_ori=copy.deepcopy(preds)
		f1,precision,recall=f1_utils.get_f1(preds,labels,mode='at')	
		outs=[]
		outs+=['[ result audio tagging %s f1 : %f, precision : %f, recall : %f ]'
						%(mode,f1,precision,recall)]

		for o in outs:
			LOG.info(o)
		self.save_str(result_path,outs,is_add)
		return preds_ori


	def save_sed_result(self,sed_preds={}):
		preds_out={}
		preds_out['vali']=self.save_sed(mode='vali',
					sed_preds=sed_preds,is_add=False)
		preds_out['test']=self.save_sed(mode='test',
					sed_preds=sed_preds,is_add=True)
		return preds_out

	def save_sed(self,mode='test',sed_preds={},is_add=False):

		model_path=self.best_model_path
		result_dir=self.result_dir
		model_name=self.model_name

		data_loader=self.data_loader
		f1_utils=self.utils

		result_path=os.path.join(result_dir,model_name+'_sed.txt')
		detail_result_path=os.path.join(result_dir,model_name+'_detail_sed.txt')
		preds_csv_path=os.path.join(result_dir,model_name+'_%s_preds.csv'%mode)
		preds,frame_preds=self.test(mode,'sed',sed_preds)
		ori_frame_preds=copy.deepcopy(frame_preds)
		outs=[]


		if mode=='vali':
			
			lst,csv=data_loader.get_vali()
		else:
			lst,csv=data_loader.get_test()
		f1_utils.set_vali_csv(lst,csv)
		segment_based_metrics,event_based_metrics,f1,er=f1_utils.get_f1(preds,frame_preds,mode='sed')
		seg_event=[segment_based_metrics,event_based_metrics]
		seg_event_str=['segment_based','event_based']
		for i,u in enumerate(seg_event):
			re=u.results_class_wise_average_metrics()
			f1=re['f_measure']['f_measure']
			er=re['error_rate']['error_rate']
			pre=re['f_measure']['precision']
			recall=re['f_measure']['recall']
			dele=re['error_rate']['deletion_rate']
			ins=re['error_rate']['insertion_rate']
			outs+=['[ result sed %s %s macro f1 : %f, er : %f, pre : %f, recall : %f, deletion : %f, insertion : %f ]'%(mode,seg_event_str[i],f1,er,pre,recall,dele,ins)]

		for o in outs:
			LOG.info(o)
		self.save_str(result_path,outs,is_add)
		for u in seg_event:
			self.save_str(detail_result_path,[u.__str__()],is_add)
			is_add=True
		shutil.copyfile(f1_utils.preds_path,preds_csv_path)
		#LOG.info(result)
		preds=np.reshape(preds,[preds.shape[0],1,preds.shape[1]])
		return ori_frame_preds*preds

	def save_str(self,text,content,is_add=False):
		content+=['']
		if is_add:
			a='a'
		else:
			a='w' 
		with open(text,a) as f:
			f.writelines('\n'.join(content))
	
