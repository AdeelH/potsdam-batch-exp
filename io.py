import pickle
import boto3


class LocalIoHandler(object):
	def __init__(self, root='.'):
		self.root = root

	def _to_local_path(self, path):
		return f'{self.root}/{path}'

	def load_pickled_file(self, path):
		with open(self._to_local_path(path), 'rb') as f:
			data = pickle.load(f)
		return data

	def save_pickled_file(self, path, data):
		with open(self._to_local_path(path), 'wb') as f:
			pickle.dump(data, f)


	def append_to_file(self, path, s):
		with open(self._to_local_path(path), 'a+') as f:
	        f.write(f'{s}\n')
	        f.flush()

    def save_img(self, fig, path):
    	fig.savefig(self._to_local_path(path))

	def save_model(self, model, path, info={}):
	    state = {
	        'model': model.state_dict(),
	        'info': info
	    }
	    torch.save(state, path)

	def save_log(self, path, log):
		self.save_pickled_file(path, log)

	def save_log_str(self, path, log_str):
		self.append_to_file(path, log_str)


class S3IoHandler(object):
	def __init__(self, local_root, s3_bucket, s3_root):
		self.s3 = boto3.client('s3')
		self.s3_root = s3_root
		self.s3_bucket = s3_bucket
		self.local_root = local_root
		self.local_io_handler = LocalIoHandler(local_root)

	def _to_s3_path(self, path):
		return f'{self.s3_root}/{path}'

	def _to_local_path(self, path):
		return f'{self.local_root}/{path}'

	def load_pickled_file(self, path):
		s3_path = self._to_s3_path(path)
		self.s3.download_file(self.s3_bucket, s3_path, path)
		return self.local_io_handler.load_pickled_file(path)

	def save_pickled_file(self, path, data):
		local_path = self._to_local_path(path)
		self.local_io_handler.save_pickled_file(path, data)
	    self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def append_to_file(self, path, s):
		local_path = self._to_local_path(path)
		s3_path = self._to_s3_path(path)
		self.local_io_handler.append_to_file(path, s)
	    self.s3.upload_file(local_path, self.s3_bucket, s3_path)

    def save_img(self, fig, path):
		local_path = self._to_local_path(path)
		s3_path = self._to_s3_path(path)
    	self.local_io_handler(fig, path)
	    self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def save_model(self, model, path, info={}):
		local_path = self._to_local_path(path)
		s3_path = self._to_s3_path(path)
	    self.local_io_handler.save_model(model, path, info)
	    self.s3.upload_file(local_path, self.s3_bucket, s3_path)

	def save_log(self, path, log):
		self.save_pickled_file(path, log)

	def save_log_str(self, path, log_str):
		self.append_to_file(path, log_str)

