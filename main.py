import os
import numpy as np
import random
import shutil
import sys
from src import trainer
from keras import backend as K
import tensorflow as tf
import argparse
from src.Logger import LOG
from keras.layers import Input,concatenate,GaussianNoise
from keras.models import load_model,Model
import shutil
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def supervised_train(task_name,sed_model_name,augmentation):
	LOG.info('-----------config preparation for %s-----------'%sed_model_name)
	train_sed=trainer.trainer(task_name,sed_model_name,False)
	creat_model_sed=train_sed.model_struct.graph()
	LEN=train_sed.data_loader.LEN
	DIM=train_sed.data_loader.DIM
	inputs=Input((LEN,DIM))
	if augmentation:
		inputs_t=GaussianNoise(0.15)(inputs)
	else:
		inputs_t=inputs
	outs=creat_model_sed(inputs_t,False)
	models=Model(inputs,outs)
	LOG.info('------------start training------------')
	train_sed.train(extra_model=models,train_mode='supervised')
	train_sed.save_at_result()
	train_sed.save_sed_result()

def semi_train(task_name,sed_model_name,at_model_name,augmentation):
	LOG.info('-----------config preparation for %s-----------'%at_model_name)
	train_sed=trainer.trainer(task_name,sed_model_name,False)
	LOG.info('-----------config preparation for %s-----------'%sed_model_name)
	train_at=trainer.trainer(task_name,at_model_name,False)
	#combine two models
	creat_model_at=train_at.model_struct.graph()
	creat_model_sed=train_sed.model_struct.graph()
	LEN=train_sed.data_loader.LEN
	DIM=train_sed.data_loader.DIM	
	inputs=Input((LEN,DIM))
	if augmentation:
		at_inputs=GaussianNoise(0.15)(inputs)
	else:
		at_inputs=inputs
	at_out=creat_model_at(at_inputs,False)
	sed_out=creat_model_sed(inputs,False)
	out=concatenate([at_out,sed_out],axis=-1)
	models=Model(inputs,out)
	LOG.info('------------start training------------')	
	train_sed.train(models)
	shutil.copyfile(train_sed.best_model,train_at.best_model) 
	LOG.info('------------result of %s------------'%at_model_name)
	train_at.save_at_result()
	LOG.info('------------result of %s------------'%sed_model_name)
	train_sed.save_at_result()
	train_sed.save_sed_result()

def test2(task_name,model_name):
	train=trainer.trainer(task_name,sed_model_name,True)
	model_lst=['-1','-3','-5']
	pred_at={}
	pred_sed={}
	for m in model_lst:
		
		check='exp/TEST/sed_with_cATP-DF%s/sed_test.npy'%m
		if not os.path.exists(check):
			train_m=trainer.trainer(task_name,sed_model_name+m,True)
			train.best_model=train_m.best_model
			pred_at_m=train.save_at_result()
			pred_sed_m=train.save_sed_result()
			for mode in ['vali','test']:
				at_path='exp/TEST/sed_with_cATP-DF%s/at_%s.npy'%(m,mode)
				sed_path='exp/TEST/sed_with_cATP-DF%s/sed_%s.npy'%(m,mode)
				pred_at_m[mode][pred_at_m[mode]>=0.5]=1
				pred_at_m[mode][pred_at_m[mode]<0.5]=0
				mask=np.reshape(pred_at_m[mode],
			[pred_at_m[mode].shape[0],1,pred_at_m[mode].shape[1]])
				np.save(at_path,pred_at_m[mode])
				np.save(sed_path,pred_sed_m[mode])
				
		for mode in ['vali','test']:
			at_path='exp/TEST/sed_with_cATP-DF%s/at_%s.npy'%(m,mode)
			sed_path='exp/TEST/sed_with_cATP-DF%s/sed_%s.npy'%(m,mode)
			pred_at_m=np.load(at_path)
			pred_sed_m=np.load(sed_path)
			if not mode in pred_at:
				pred_at[mode]=pred_at_m
				pred_sed[mode]=pred_sed_m
			else:
				pred_at[mode]+=pred_at_m
				pred_sed[mode]+=pred_sed_m
	preds={}
	for mode in pred_at:
		mask=np.reshape(pred_at[mode],
			[pred_at[mode].shape[0],1,pred_at[mode].shape[1]])
		mask[mask==0]=1
		pred_sed[mode]/=mask
		pred_at[mode]=pred_at[mode]/len(model_lst)
		preds[mode]=[pred_at[mode],pred_sed[mode]]

	train.save_sed_result(preds)

def test(task_name,model_name):
	train=trainer.trainer(task_name,model_name,True)
	train.save_at_result()
	train.save_sed_result()

def bool_convert(value):
	if value=='true' or value=='True':
		value=True
	elif value=='false' or value=='False':
		value=False
	else:
		assert True
	return value

if __name__=='__main__':
	LOG.info('Disentangled feature')
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-t', '--task_name', 
			dest='task_name',
			help='task name')

	parser.add_argument('-s', '--sed_model_name', dest='sed_model_name',
		help='start with sed for sound event detection; eg. sed_with_cATP-DF')
	parser.add_argument('-a', '--at_model_name', dest='at_model_name',
		help='start with at for audio tagging; eg. at_with_cATP-DF')
	parser.add_argument('-md', '--mode', dest='mode',
		help='train or test')
	parser.add_argument('-u', '--test_unlabel', dest='test_unlabel', 
		default=False,
		help='whether to test unlabel data')
	parser.add_argument('-n', '--augmentation', dest='augmentation',
		default=True,
		help='whether to use augmentation')
	parser.add_argument('-p', '--semi_supervised', dest='semi_supervised',
		default=True,
		help='whether to use unlabel data')
	f_args = parser.parse_args()

	task_name=f_args.task_name
	sed_model_name=f_args.sed_model_name
	at_model_name=f_args.at_model_name
	mode=f_args.mode
	test_unlabel=f_args.test_unlabel
	semi_supervised=f_args.semi_supervised
	augmentation=f_args.augmentation
	test_unlabel=bool_convert(test_unlabel)
	augmentation=bool_convert(augmentation)
	semi_supervised=bool_convert(semi_supervised)

	if task_name is None or sed_model_name is None or (semi_supervised and at_model_name is None) or mode is None:
		assert LOG.info('try add --help to get usage')
	if (not str.startswith(sed_model_name,'sed') and  not str.startswith(sed_model_name,'at')) or (semi_supervised and (str.startswith(at_model_name,'sed') and  not str.startswith(at_model_name,'at'))):
		assert LOG.info('model name should be started with [sed] or [at]; try add --help to get usage')
	if not mode=='test' and not mode=='train':
		assert LOG.info('mode should be [train] or [test] ; try add --help to get usage')
	LOG.info( 'task name: {}'.format(task_name))
	LOG.info( 'sed model name: {}'.format(sed_model_name))
	LOG.info( 'at model name: {}'.format(at_model_name))
	LOG.info( 'mode: {}'.format(mode))
	LOG.info( 'test unlabel: {}'.format(test_unlabel))
	LOG.info( 'semi_supervised: {}'.format(semi_supervised))
	
	if test_unlabel:
		test_unlabel(task_name,sed_model_name)
	elif mode=='train':
		if semi_supervised:
			semi_train(task_name,sed_model_name,at_model_name,
				augmentation)
		else:
			supervised_train(task_name,sed_model_name,augmentation)		
	else:
		test(task_name,sed_model_name)
		

