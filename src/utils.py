from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import dcase_util
import os
import numpy as np
import configparser
import scipy
from src.evaluation import sound_event_eval
from src.evaluation import scene_eval
import src.evaluation.TaskAEvaluate as taskAEvaluate
import copy
class utils(object):
	def __init__(self,conf_dir,
			exp_dir,
			metric,
			ave,
			label_lst):
		self.conf_dir=conf_dir
		self.metric=metric
		self.ave=ave
		self.label_lst=label_lst
		#self.init_utils_conf()
		self.evaluation_path=os.path.join(exp_dir,'evaluation')	
		self.init_dirs(self.evaluation_path)
		self.evaluation_ests=os.path.join(self.evaluation_path,'ests')
		self.preds_path=os.path.join(self.evaluation_path,'preds.csv')
		self.init_dirs(self.evaluation_ests)
		self.evaluation_refs=os.path.join(self.evaluation_path,'refs')
		self.init_dirs(self.evaluation_refs)

	def init_dirs(self,path):
		if not os.path.exists(path):
			os.mkdir(path)
			
	def init_csv(self,csvs,flag=True):
		if flag:
			root=self.evaluation_refs
		else:
			root=self.evaluation_ests
		result=self.fomat_lst(csvs)
		self.fomat_csv(result,root)


	def init_utils_conf(self,n):
		CLASS=n
		conf_dir=self.conf_dir
		utils_cfg_path=os.path.join(conf_dir,'utils.cfg')

		assert os.path.exists(utils_cfg_path)
		config=configparser.ConfigParser()
		config.read(utils_cfg_path)
		conf=config['sed']
		win_len=conf['win_len']
		if not win_len=='auto':
			self.win_lens=[int(conf['win_len'])]*CLASS



	def fomat_lst(self,tests):
		tests=[t.rstrip().split('\t') for t in tests]
		result={}
		cur=0
		for i,t in enumerate(tests):
			f=str.replace(t[0],'.wav','')
			if f not in result:
				result[f]=[]
			if len(t)>1:
				result[f]+=[[t[1],t[2],t[3]]]
			else:
				cur+=1
		return result

	def fomat_at_2017(self,tests,lst,root,CLASS,is_ref=False):
		t_len=tests.shape[0]
		label_lst=self.label_lst
		if is_ref:
			path=os.path.join(root,'refs.csv')
		else:
			path=os.path.join(root,'ests.csv')
		result=[]
		for i in range(t_len):
			flags=True
			for j in range(CLASS):
				if tests[i][j]==1:
					result+=['%s.wav\t0\t10\t%s'
						%(lst[i],label_lst[j])]
					flags=False
			if flags:
				result+=['%s.wav'%lst[i]]
		with open(path,'w') as f:
			f.writelines('\n'.join(result))
		return path
		
	def fomat_at(self,tests,lst,root,CLASS,is_ref=True):
		t_len=tests.shape[0]
		label_lst=self.label_lst
		file_lst=[]
		if is_ref:
			path=os.path.join(root,'refs')
		else:
			path=os.path.join(root,'ests')
		for i in range(t_len):
			result=[]
			for j in range(CLASS):
				if tests[i][j]==1:
					result+=['%s.wav\t%s'
						%(lst[i],label_lst[j])]
			if len(result)==0:
				result=['%s.wav\t'%lst[i]]
			with open(os.path.join(path,'at_%s.txt'%lst[i]),'w') as f:
				f.writelines('\n'.join(result))
			if not is_ref:
				file_lst+=[{'reference_file':'refs/at_%s.txt'%lst[i],
					'estimated_file':'ests/at_%s.txt'%lst[i]}]
		if not is_ref:
			return file_lst
					
	def fomat_csv(self,tests,root):
		for t in tests:
			fname=os.path.join(root,t)
			with open(fname+'.txt','w') as f:
				result=[]
				for k in tests[t]:
					if len(k)>1:
						result+=['%s\t%s\t%s'%(k[0],k[1],k[2])]
				result='\n'.join(result)
				f.writelines(result)
			


	def set_vali_csv(self,lst,csv):
		self.lst=lst
		self.csv=csv

	def get_f1(self,preds,labels,mode='at'):
		lst,csv=self.get_vali_lst()
		preds=preds[:len(lst)]
		preds[preds>=0.5]=1
		preds[preds<0.5]=0

		evaluation_path=self.evaluation_path
			
		if mode=='at':
			labels=labels[:len(lst)]	
			ave=self.ave
			CLASS=labels.shape[-1]
			if ave=='acc':
				self.fomat_at(labels,lst,evaluation_path,
								CLASS,True)
				f_lst=self.fomat_at(preds,lst,evaluation_path,
								CLASS,False)
				result=scene_eval.main(evaluation_path,f_lst)
				re=result.results_overall_metrics()
				acc=re['accuracy']
				F1=acc
				precision=acc
				recall=acc
			elif ave=='F1':
				refs=self.fomat_at_2017(labels,lst,
					evaluation_path,CLASS,is_ref=True)
				ests=self.fomat_at_2017(preds,lst,
						evaluation_path,CLASS)	
				F1,precision,recall=taskAEvaluate.evaluateMetrics(
							refs,ests)
			elif ave=='class_wise_F1' or ave=='overall_F1':
				TP=(labels+preds==2).sum(axis=0)
				FP=(labels-preds==-1).sum(axis=0)
				FN=(labels-preds==1).sum(axis=0)
				if ave=='overall_F1':
					TP=np.sum(TP)
					FP=np.sum(FP)
					FN=np.sum(FN)

				TFP=TP+FP
				if ave=='overall_F1':
					if TFP==0:
						TFP=1
				else:	
					TFP[TFP==0]=1
				precision=TP/TFP
				TFN=TP+FN
				if ave=='overall_F1':
					if TFN==0:
						TFN=1
				else:
					TFN[TFN==0]=1
				recall=TP/TFN
				pr=precision + recall
				if ave=='overall_F1':
					if pr==0:
						pr=1
				else:
					pr[pr==0]=1
				F1=2*precision*recall/pr
				if ave=='overall_F1':
					return F1,precision,recall
				
			return np.mean(F1),np.mean(precision),np.mean(recall)

		elif mode=='sed':
			segment_based_metrics,event_based_metrics,f1,er=self.get_sed_result(preds,labels)
			return segment_based_metrics,event_based_metrics,f1,er
			
		
	def get_vali_lst(self):
		return self.lst,self.csv

	def set_win_lens(self,win_lens):
		self.win_lens=win_lens
		self.init_utils_conf(len(win_lens))

	def get_predict_csv(self,results):
		outs=[]
		for re in results:
			flag=True
			for line in results[re]:
				outs+=['%s.wav\t%s\t%s\t%s'%(
					re,line[0],line[1],line[2])]
				flag=False
			if flag:
				outs+=['%s.wav'%re]
		with open(self.preds_path,'w') as f:
			f.writelines('\n'.join(outs))
		return outs
	
	def get_sed_result(self,preds,frame_preds):

		lst,csv=self.get_vali_lst()
		label_lst=self.label_lst
		win_lens=self.win_lens

		result=[]
		CLASS=frame_preds.shape[-1]
		top_LEN=frame_preds.shape[1]
		hop_len=10.0/top_LEN
		print(hop_len)
		decision_encoder=dcase_util.data.DecisionEncoder(
			label_list=label_lst)
		frame_preds=frame_preds[:len(lst)]

		shows=[]
		result={}
		file_lst=[]

		for i in range(len(lst)):
			pred=preds[i]
			frame_pred=frame_preds[i]
			for j in range(CLASS):
				if pred[j]==0:
					frame_pred[:,j]*=0
					
				frame_pred[:,j]=scipy.ndimage.filters.median_filter(frame_pred[:,j], (win_lens[j]))	
			frame_decisions=dcase_util.data.ProbabilityEncoder()\
				.binarization(
					probabilities=frame_pred,
					binarization_type='global_threshold',
					time_axis=0)
			for j in range(CLASS):
				frame_decisions[:,j]=scipy.ndimage.filters.median_filter(frame_decisions[:,j], (win_lens[j]))
			flags=True
			if lst[i] not in result:
				result[lst[i]]=[]
				file_lst+=[{'reference_file':'refs/%s.txt'%lst[i],
					'estimated_file':'ests/%s.txt'%lst[i]}]
			for j in range(CLASS):
				estimated_events=decision_encoder\
						.find_contiguous_regions(
					activity_array=frame_decisions[:,j])
				
				for [onset, offset] in estimated_events:
					result[lst[i]]+=[[str(onset*hop_len),
							str(offset*hop_len),
							label_lst[j]]]
		self.get_predict_csv(result)
		self.fomat_csv(result,self.evaluation_ests)
		segment_based_metrics,event_based_metrics=sound_event_eval.main(
			self.evaluation_path,file_lst)
		if self.metric=='EventBasedMetrics':
			result=event_based_metrics
			re=result.results_class_wise_average_metrics()
		else:
			result=segment_based_metrics
			re=result.results_overall_metrics()
		f1=re['f_measure']['f_measure']
		er=re['error_rate']['error_rate']
		return segment_based_metrics,event_based_metrics,f1,er

		
