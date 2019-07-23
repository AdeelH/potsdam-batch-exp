import pickle
import boto3
import os
import torch

class LocalIoHandler(object):
	def __init__(self, root='.'):
		self.root = root
		os.makedirs(root, exist_ok=True)
		os.makedirs(self.to_local_path('data'), exist_ok=True)
		os.makedirs(self.to_local_path('model'), exist_ok=True)
		os.makedirs(self.to_local_path('checkpoints'), exist_ok=True)
		os.makedirs(self.to_local_path('best_model'), exist_ok=True)
		os.makedirs(self.to_local_path('visualizations'), exist_ok=True)

	def to_local_path(self, path):
		return f'{self.root}/{path}'

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

	def save_img(self, fig, path):
		fig.savefig(self.to_local_path(path))

	def save_model(self, model, path, info={}):
		state = {
			'model': model.state_dict(),
			'info': info
		}
		torch.save(state, self.to_local_path(path))

	def load_model_weights(self, path, model):
	    state = torch.load(self.to_local_path(path))
	    print(state['info'])
	    model.load_state_dict(state['model'])

	def save_log(self, path, log):
		self.save_pickled_file(path, log)

	def save_log_str(self, path, log_str):
		self.append_to_file(f'logs/{path}', log_str)


class S3IoHandler(object):
	def __init__(self, local_root, s3_bucket, s3_root):
		self.s3 = boto3.client('s3')
		self.s3_root = s3_root
		self.s3_bucket = s3_bucket
		self.local_root = local_root
		self.local_io_handler = LocalIoHandler(local_root)
		self.to_local_path = self.local_io_handler.to_local_path

	def to_s3_path(self, path):
		return f'{self.s3_root}/{path}'

	def load_pickled_file(self, s3_path, tgt_path, force_download=False):
		local_path = self.to_local_path(tgt_path)
		if force_download or (not os.path.isfile(local_path)):
			self.s3.download_file(self.s3_bucket, s3_path, local_path)
		return self.local_io_handler.load_pickled_file(tgt_path)

	def save_pickled_file(self, path, data):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler.save_pickled_file(path, data)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def append_to_file(self, path, s):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler.append_to_file(path, s)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def save_img(self, fig, path):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler(fig, path)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def save_model(self, model, path, info={}):
		local_path = self.to_local_path(path)
		s3_path = self.to_s3_path(path)
		self.local_io_handler.save_model(model, path, info)
		self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def load_model_weights(self, path, model):
	    state = torch.load(self.to_local_path(path))
	    print(state['info'])
	    model.load_state_dict(state['model'])

	def save_log(self, path, log):
		self.save_pickled_file(path, log)

	def save_log_str(self, path, log_str):
		self.append_to_file(path, log_str)

