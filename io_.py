import pickle
import boto3
import os
import torch
import numpy as np
from glob import glob
import re

class LocalIoHandler(object):
	def __init__(self, root='.'):
		self.root = root
		os.makedirs(root, exist_ok=True)
		os.makedirs(self.to_local_path('data'), exist_ok=True)
		os.makedirs(self.to_local_path('models'), exist_ok=True)
		os.makedirs(self.to_local_path('checkpoints'), exist_ok=True)
		os.makedirs(self.to_local_path('best_model'), exist_ok=True)
		os.makedirs(self.to_local_path('visualizations'), exist_ok=True)

	def to_local_path(self, path):
		return f'{self.root}/{path}'

	def file_exists(self, path):
		return os.path.isfile(self.to_local_path(path))

	def load_pickled_file(self, path):
		with open(self.to_local_path(path), 'rb') as f:
			data = pickle.load(f)
		return data

	def save_pickled_file(self, path, data):
		with open(self.to_local_path(path), 'wb') as f:
			pickle.dump(data, f)

	def append_to_file(self, path, s):
		with open(self.to_local_path(path), 'a+') as f:
			f.write(f'{s}\n')
			f.flush()

	def save_img(self, fig, path, ext='.png'):
		fig.savefig(f'{self.to_local_path(path)}{ext}')

	def save_model(self, model, path, info={}):
		state = {
			'model': model.state_dict(),
			'info': info
		}
		torch.save(state, self.to_local_path(path))

	def save_checkpoint(self, model, optimizer, sched, path, info={}):
		state = {
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'sched': sched.state_dict(),
			'info': info
		}
		torch.save(state, self.to_local_path(path))

	def load_saved_model(self, path):
		state = torch.load(self.to_local_path(path))
		return state

	def load_model_weights(self, model, path):
		state = self.load_saved_model(path)
		model.load_state_dict(state['model'])

	def save_log(self, path, log):
		self.save_pickled_file(path, log)

	def save_log_str(self, path, log_str):
		self.append_to_file(f'logs/{path}', log_str)

	def checkpoint_exists(self):
		chkpt_fnames = glob(f'{self.to_local_path("checkpoints")}/epoch_*')
		return len(chkpt_fnames) > 0

	def load_latest_checkpoint(self):
		chkpt_fnames = glob(f'{self.to_local_path("checkpoints")}/epoch_*')
		regex = re.compile(r'epoch_(\d+).*')
		latest_chkpt = np.argmax([int(regex.match(f).group(1)) for f in chkpt_fnames])
		chkpt = self.load_saved_model(f'checkpoints/{chkpt_fnames[latest_chkpt]}')
		return chkpt


class S3IoHandler(object):
	def __init__(self, local_root, s3_bucket, s3_root):
		self.s3 = boto3.client('s3')
		self.s3_root = s3_root
		self.s3_bucket = s3_bucket
		self.local_root = local_root
		self.local_io_handler = LocalIoHandler(local_root)
		self.to_local_path = self.local_io_handler.to_local_path
		self.file_exists = self.local_io_handler.file_exists

	def to_s3_path(self, path):
		return f'{self.s3_root}/{path}'

	def load_pickled_file(self, s3_path, tgt_path, force_download=False):
		if force_download or (not self.file_exists(tgt_path)):
			local_path = self.to_local_path(tgt_path)
			self.s3.download_file(self.s3_bucket, s3_path, local_path)
		return self.local_io_handler.load_pickled_file(tgt_path)

	def save_pickled_file(self, path, data):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler.save_pickled_file(path, data)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def upload_file(self, src_path, tgt_path):
		local_path = self.to_local_path(src_path)
		s3_path = self.to_s3_path(tgt_path)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def append_to_file(self, path, s):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler.append_to_file(path, s)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def save_img(self, fig, path, ext='.png'):
		self.local_io_handler.save_img(fig, path, ext=ext)
		
		local_path = f'{self.to_local_path(path)}{ext}'
		s3_path = f'{self.to_s3_path(path)}{ext}'
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def save_model(self, model, path, info={}):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler.save_model(model, path, info)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def save_checkpoint(self, model, optimizer, sched, path, info={}):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler.save_checkpoint(model, optimizer, sched, path, info)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def load_saved_model(self, s3_path, path, force_download=False):
		if force_download or (not self.file_exists(path)):
			local_path = self.to_local_path(path)
			self.s3.download_file(self.s3_bucket, s3_path, local_path)
		return self.local_io_handler.load_saved_model(path)

	def load_model_weights(self, model, s3_path, tgt_path, force_download=False):
		if force_download or (not self.file_exists(tgt_path)):
			local_path = self.to_local_path(tgt_path)
			self.s3.download_file(self.s3_bucket, s3_path, local_path)
		self.local_io_handler.load_model_weights(model, tgt_path)

	def save_log(self, path, log):
		self.save_pickled_file(path, log)

	def save_log_str(self, path, log_str):
		self.append_to_file(path, log_str)

	def ls(self, path):
		res = self.s3.list_objects(Bucket=self.s3_bucket, Prefix=path)
		if 'Contents' not in res:
			return []
		fnames = [c['Key'] for c in res['Contents']]
		return fnames

	def checkpoint_exists(self):
		chkpt_fnames = self.ls(self.to_s3_path(f'checkpoints/epoch_'))
		return len(chkpt_fnames) > 0

	def load_latest_checkpoint(self):
		chkpt_fnames = self.ls(self.to_s3_path(f'checkpoints/epoch_'))
		regex = re.compile(r'.*epoch_(\d+).*')
		latest_chkpt = np.argmax([int(regex.match(f).group(1)) for f in chkpt_fnames])
		chkpt = self.load_saved_model(chkpt_fnames[latest_chkpt], 'latest_checkpoint')
		return chkpt
