import argparse
import multiprocessing
import os
import configparser
import librosa
import scipy
import numpy as np

eps=np.spacing(1)

class feature_extractor(object):
	def __init__(self,conf_path):
		self.conf_path=conf_path
		self.init_extractor_conf()

	def init_extractor_conf(self):
		conf_path=self.conf_path
		assert os.path.exists(path)
		config=configparser.ConfigParser()
		config.read(path)
		assert 'feature' in config.sections()
		feature_cfg=config['feature']

		self.n_fft=int(feature_cfg['n_fft'])
		self.n_mels=int(feature_cfg['n_mels'])
		f_max=feature_cfg['f_max']
		if not f_max=='max':
			f_max=int(f_max)
		f_min=int(feature_cfg['f_min'])
		assert f_max>f_min
		self.f_max=f_max
		self.f_min=f_min
		self.LEN=int(feature_cfg['LEN'])

		hop_length=float(feature_cfg['hop_length'])
		win_length=float(feature_cfg['win_length'])
		sr=int(feature_cfg['sr'])
		self.hop_length=hop_length
		self.win_length=win_length
		self.sr=sr
		self.win_length_second=int(sr*win_length)
		self.hop_length_second=int(sr*hop_length)




	def get_feature(self,input_file,output_file):
		sr=self.sr
		n_fft=self.n_fft
		n_mels=self.n_mels
		f_min=self.f_min
		f_max=self.f_max
		win_length_second=self.win_length_second
		hop_length_second=self.hop_length_second
		LEN=self.LEN

		y,_=librosa.load(input_file,sr=sr)
		win=scipy.signal.hann(win_length_second, sym=False)
		mel_basis=librosa.filters.mel(sr=sr,n_fft=n_fft,n_mels=n_mels,
			fmin=f_min,fmax=f_max,htk=False)
		spectrogram=np.abs(librosa.stft(y+eps,
			n_fft=n_fft,
			win_length=win_length_second,
			hop_length=hop_length_second,
			center=True,
			window=win))
	
		mel_spectrum=np.dot(mel_basis, spectrogram)
		log_mel_spectrum=np.log(mel_spectrum+eps)
		feature=np.transpose(log_mel_spectrum)
		flen=int(sr*10/hop_length_second+1)
		np.save(output_file,feature)
		if feature.shape[0]<flen:
			new_feature=np.zeros([flen,feature.shape[1]])
			new_feature[:feature.shape[0]]=feature
		else:
			new_feature=feature[:flen]	
	
		if not LEN==flen:
			squeeze=(flen-LEN)//2
		if squeeze<0:
			final_feature=np.zeros([LEN,new_feature.shape[1]])
			final_feature[-squeeze:LEN+squeeze]=new_feature
		else:
			lsq=squeeze
			rsq=flen-LEN-squeeze
			final_feature=new_feature[lsq:flen-rsq]
		np.save(output_file,final_feature)
		return final_feature


	def get_feature_for_single_lst(self,lst,wav_dir,feature_dir,id):
		for f in lst:
			input_file=os.path.join(wav_dir,f+'.wav')
			output_file=os.path.join(feature_dir,f)
			print('strat processing %d : %s'%(id,f))
			if os.path.exists(output_file+'.npy'):
				print('process %d : %s exists'%(id,f))
				continue
			self.get_feature(input_file,output_file)
			print('process %d : %s done'%(id,f))
	
	def get_feature_for_lst(self,lst,wav_dir,feature_dir,processes):
		with open(lst) as f:
			lsts=f.readlines()
		lsts=[f.rstrip() for f in lsts]
		f_per_processes=(len(lsts)+processes-1)//processes
		for i in range(processes):
			st=f_per_processes*i
			ed=st+f_per_processes
			if st>=len(lsts):
				break
			if ed>len(lsts):
				ed=len(lsts)
			sub_lsts=lsts[st:ed]
			p=multiprocessing.Process(
				target=self.get_feature_for_single_lst,
				args=(sub_lsts,wav_dir,feature_dir,i+1))

			p.start()
			print('process %d start'%(i+1))


def extract_feature(wav_lst,wav_dir,feature_dir,feature_cfg,processes):
	fextractor=feature_extractor(feature_cfg)
	fextractor.get_feature_for_lst(wav_lst,wav_dir,feature_dir,processes)
	



if __name__=='__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-l', '--wav_lst',dest='wav_lst',
		help='the list of audios')

	parser.add_argument('-w', '--wav_dir', dest='wav_dir',
		help='the audio dir')
	parser.add_argument('-f', '--feature_dir', dest='feature_dir',
		help='the ouput feature dir')
	parser.add_argument('-c', '--feature_cfg', dest='feature_cfg',
		help='the config of featrue extraction')
	parser.add_argument('-p', '--processes', dest='processes',
		help='the number of processes')

	f_args = parser.parse_args()

	wav_lst=f_args.wav_lst
	wav_dir=f_args.wav_dir
	feature_dir=f_args.feature_dir
	feature_cfg=f_args.feature_cfg
	processes=int(f_args.processes)

	paths=[wav_lst,wav_dir,feature_dir,feature_cfg]
	
	for path in paths:
		print(path)
		assert os.path.exists(path)

	extract_feature(wav_lst,wav_dir,feature_dir,feature_cfg,processes)
	
