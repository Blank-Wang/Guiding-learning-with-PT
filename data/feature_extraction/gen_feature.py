import argparse
import multiprocessing
import os
import configparser
import librosa
import scipy
import numpy as np
eps=np.spacing(1)
def get_frame(data,window_length,hop_length):
        num_samples = data.shape[0]
        num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
        shape = (num_frames, window_length) + data.shape[1:]
        strides = (data.strides[0] * hop_length,) + data.strides
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def get_feature(input_file,output_file,
		sr,n_fft,hop_length,win_length,n_mels,f_min,f_max,LEN):
	y,sr=librosa.load(input_file,sr=sr)
	win_length=int(sr*win_length)
	hop_length=int(sr*hop_length)
	if f_max=='max':
		f_max=sr/2
	win=scipy.signal.hann(win_length, sym=False)
	mel_basis=librosa.filters.mel(sr=sr,n_fft=n_fft,n_mels=n_mels,
		fmin=f_min,fmax=f_max,htk=False)
	spectrogram=np.abs(librosa.stft(y+eps,n_fft=n_fft,
			win_length=win_length,
			hop_length=hop_length,
			center=True,
			window=win))
	
	mel_spectrum=np.dot(mel_basis, spectrogram)
	log_mel_spectrum=np.log(mel_spectrum+eps)
	feature=np.transpose(log_mel_spectrum)
	flen=int(sr*10/hop_length+1)
	np.save(output_file,feature)
	if feature.shape[0]<flen:
		new_feature=np.zeros([flen,feature.shape[1]])
		new_feature[:feature.shape[0]]=feature
	else:
		new_feature=feature[:flen]	
	if not LEN==flen:
		squeeze=(flen-LEN)//2
		if squeeze<0:
			new_feature=np.zeros([LEN,new_feature.shape[1]])
			new_feature[-squeeze:flen-squeeze]=new_feature
		else:
			lsq=squeeze
			rsq=squeeze+(flen-LEN)-squeeze*2
			new_feature=new_feature[lsq:flen-rsq]
	#sr=1.0 / (hop_length/sr)
	#new_feature=get_frame(new_feature,int(sr*0.96),int(sr*0.96))
	np.save(output_file,new_feature)
	return new_feature

def get_feature_config(path):
	assert os.path.exists(path)
	config=configparser.ConfigParser()
	config.read(path)
	assert 'feature' in config.sections()
	feature_cfg=config['feature']
	n_fft=int(feature_cfg['n_fft'])
	n_mels=int(feature_cfg['n_mels'])
	hop_length=float(feature_cfg['hop_length'])
	f_min=int(feature_cfg['f_min'])
	sr=int(feature_cfg['sr'])
	win_length=float(feature_cfg['win_length'])
	f_max=feature_cfg['f_max']
	LEN=int(feature_cfg['LEN'])
	if not f_max=='max':
		f_max=int(f_max)
		assert f_max>f_min
	return sr,n_fft,hop_length,win_length,n_mels,f_min,f_max,LEN

def get_feature_for_single_lst(lst,wav_dir,feature_dir,feature_cfg,id):
	sr,n_fft,hop_length,win_length,n_mels,f_min,f_max,LEN=get_feature_config(
		feature_cfg)
	for f in lst:
		input_file=os.path.join(wav_dir,f+'.wav')
		output_file=os.path.join(feature_dir,f)
		if os.path.exists(output_file+'.npy'):
			print('process %d : %s exists'%(id,f))
			continue
		get_feature(input_file,output_file,
			sr,n_fft,hop_length,win_length,n_mels,f_min,f_max,LEN)
		print('process %d : %s'%(id,f))
	
def get_feature_for_lst(lst,wav_dir,feature_dir,feature_cfg,processes):
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
			target=get_feature_for_single_lst,
			args=(sub_lsts,wav_dir,feature_dir,feature_cfg,i+1))

		p.start()
		print('process %d start'%(i+1))
def test():
	get_feature_for_lst('wav.lst','wav','feature','feature.cfg',20)

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

	get_feature_for_lst(wav_lst,wav_dir,feature_dir,feature_cfg,processes)
	
