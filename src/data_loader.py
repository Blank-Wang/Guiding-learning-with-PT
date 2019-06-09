import os
import numpy as np
import random
import configparser

class data_loader(object):
	def __init__(self,conf_dir):
		self.conf_dir=conf_dir
		self.init_data_conf()

	def init_data_conf(self):
		conf_dir=self.conf_dir
		data_cfg_path=os.path.join(conf_dir,'data.cfg')
		assert os.path.exists(data_cfg_path)
		config=configparser.ConfigParser()
		config.read(data_cfg_path)

		assert 'path' in config.sections()
		path_cfg=config['path']
		
		self.feature_dir=path_cfg['feature_dir']
		self.label_dir=path_cfg['label_dir']
		self.train_lst=path_cfg['train_lst']
		self.vali_lst=path_cfg['vali_lst']
		self.test_lst=path_cfg['test_lst']
		self.vali_csv=path_cfg['vali_csv']
		self.test_csv=path_cfg['test_csv']
		self.win_len_csv=path_cfg['win_len_csv']

		files=[self.feature_dir,
			self.label_dir,
			self.train_lst,
			self.test_lst,
			self.vali_lst,
			self.test_csv,
			self.vali_csv,
			self.win_len_csv]
	
		for  f in files:
			assert os.path.exists(f)

		assert 'parameter' in config.sections()
		parameter_cfg=config['parameter']
		self.LEN=int(parameter_cfg['LEN'])
		self.DIM=int(parameter_cfg['DIM'])
		self.batch_size=int(parameter_cfg['batch_size'])
		self.dinsentangle_n=int(parameter_cfg['dinsentangle_n'])
		self.dinsentangle_a=float(parameter_cfg['dinsentangle_a'])
		self.ratio_for_win_len=float(parameter_cfg['ratio_for_win_len'])

		assert'events' in config.sections()
		event_cfg=config['events']
		
		self.events=event_cfg['events'].split(',')
		self.CLASS=len(self.events)
		self.ep_per_epochs=1			

	def read_lst(self,lst):
		with open(lst) as f:
			files=f.readlines()
		files=[f.rstrip() for f in files]
		f_len=len(files)
		return files,f_len
	
	def get_train(self):
		lst,_=self.read_lst(self.train_lst)
		return lst

	def get_vali(self):
		lst,_=self.read_lst(self.vali_lst)
		csv,_=self.read_lst(self.vali_csv)
		return lst,csv

	def get_test(self):
		lst,_=self.read_lst(self.test_lst)
		csv,_=self.read_lst(self.test_csv)
		return lst,csv


	def count_disentangle(self):
		n=self.dinsentangle_n
		a=self.dinsentangle_a
		lst=self.get_train()
		label_dir=self.label_dir
		desentangle=np.zeros([self.CLASS])
		for f in lst:
			path=os.path.join(label_dir,f+'.npy')
			if os.path.exists(path):
				label=np.load(path)
				if np.sum(label)==n:
					desentangle+=label
		desentangle=desentangle/np.max(desentangle)*(1-a)+a
		return desentangle	

	def count_win_len_per_class(self,top_len):
		path=self.win_len_csv
		ratio_for_win_len=self.ratio_for_win_len
		csv,clen=self.read_lst(path)
		label_cnt={}
		for event in self.events:
			label_cnt[event]={'num':0,'frame':0}
		frames_per_second=top_len/10.0
		for c in csv:
			cs=c.split('\t')
			if len(cs)<4:
				continue
			label=cs[3]
			label_cnt[label]['num']+=1
			label_cnt[label]['frame']+=(
				(float(cs[2])-float(cs[1]))*frames_per_second)
		for label in label_cnt:
			label_cnt[label]['win_len']=int(label_cnt[label]['frame']/label_cnt[label]['num'])
		out=[]
		for label in label_cnt:
			out+=[int(label_cnt[label]['win_len']*ratio_for_win_len)]
			if out[-1]==0:
				out[-1]=1
		print(out)
		return out
		
	def set_semi_parameter(self,exponent,start_epoch):
		self.exponent=exponent
		self.start_epoch=start_epoch

	def set_ep_per_epochs(self,ep_per_epochs):
		self.ep_per_epochs=ep_per_epochs
	
	def generator_train(self):
		train_lst=self.train_lst
		batch_size=self.batch_size
		feature_dir=self.feature_dir
		label_dir=self.label_dir
		files,f_len=self.read_lst(train_lst)
		random.shuffle(files)
		CLASS=self.CLASS
		LEN=self.LEN
		DIM=self.DIM
		start_epoch=self.start_epoch
		exponent=self.exponent
		ep_per_epochs=self.ep_per_epochs
		steps=(f_len*ep_per_epochs+batch_size-1)//batch_size
		def generator():
			i=0
			cur=0
			epoch=0
			step=0
			while True:
				f=files[i]
				i=(i+1)%f_len
				data_f=os.path.join(feature_dir,f+'.npy')
				assert os.path.exists(data_f)
				data=np.load(data_f)
				label_f=os.path.join(label_dir,f+'.npy')
				if os.path.exists(label_f):
					label=np.load(label_f)
					mask=np.ones([CLASS])
				else:
					label=np.zeros([CLASS])
					mask=np.zeros([CLASS])

				if cur==0:
					labels=np.zeros([batch_size,CLASS])
					masks=np.zeros([batch_size,CLASS])
					train_data=np.zeros(
						[batch_size,LEN,DIM])
				train_data[cur]=data
				labels[cur]=label
				masks[cur]=mask
				cur+=1
				if cur==batch_size:
					cur=0
					if epoch>start_epoch:
						a=1-np.power(exponent,epoch-start_epoch)
					else:
						a=0
					yield train_data,np.concatenate(
						[labels,masks,
					np.ones([batch_size,1])*a],axis=-1)
					step+=1
					if step%steps==0:
						epoch+=1
						step=0
				if i==0:
					print('[ epoch %d , a: %f ]'%(epoch,a))
					random.shuffle(files)
				#if i%(f_len//2)==0:
				#	epoch+=1

		return generator,steps

	def generator_vali(self,label_dir=None):
		return self.generator_all('vali',label_dir)

	def generator_test(self,label_dir=None):
		return self.generator_all('test',label_dir)

	def generator_weak(self,label_dir=None):
		return self.generator_all('weak',label_dir)

	def generator_unlabel(self,label_dir=None):
		gt,steps=self.generator('unlabel',label_dir)
		return gt

	def generator_all(self,mode,label_dir=None):
		gt,steps=self.generator(mode,label_dir)
		gt=gt()
		data=[]
		labels=[]
		for cnt,(X,Y) in enumerate(gt):
			data+=[X]
			labels+=[Y]
		data=np.concatenate(data)
		labels=np.concatenate(labels)
		return data,labels
		

	def generator(self,mode,label_dir=None):
		if mode=='vali':
			gen_lst=self.vali_lst
		elif mode=='test':
			gen_lst=self.test_lst
		elif mode=='weak':
			gen_lst=self.train_lst
		elif mode=='unlabel':
			gen_lst=self.unlabel_lst
		batch_size=self.batch_size
		feature_dir=self.feature_dir
		if label_dir==None:
			label_dir=self.label_dir
		files,f_len=self.read_lst(gen_lst)
		CLASS=self.CLASS
		LEN=self.LEN
		DIM=self.DIM
		def generator():
			cur=0
			for i in range(f_len):
				if i%batch_size==0:
					train_data=np.zeros([batch_size,
							LEN,DIM])
					if mode=='vali':
						tclass=CLASS*2+1
					else:
						tclass=CLASS
					labels=np.ones([batch_size,tclass])
					
				f=files[i]
				data_f=os.path.join(feature_dir,f+'.npy')
				assert os.path.exists(data_f)
				data=np.load(data_f)
				mask=np.ones([LEN,CLASS])
				label_f=os.path.join(label_dir,f+'.npy')
				if os.path.exists(label_f):
					label=np.load(label_f)
				else:
					label=np.zeros([CLASS])
				if data.shape[0]>LEN:
					data=data[:LEN]
				train_data[cur,:data.shape[0],:DIM]=data
				labels[cur,:CLASS]=label
				cur+=1
				if cur==batch_size:
					yield train_data,labels
					cur=0
			if not f_len%batch_size==0:
				yield train_data,labels


		steps=(f_len+batch_size-1)//batch_size
		return generator,steps
