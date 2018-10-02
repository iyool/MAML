import tensorflow as tf
import numpy as np
import datetime
from waveform_data import WavedFormData
import os
import matplotlib.pyplot as plt
from scipy import signal

class MAML_Sinusoid(object):

	def __init__(self, hidden_dims = [40,40], input_dim = 1, output_dim = 1, meta_batch_size = 32):
		self.hidden_dims = hidden_dims
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.task_lr = 0.01
		self.meta_lr = 0.01
		self.update_per_task = 5
		self.samples_per_task = 10
		self.vars = self.generate_var()
		self.meta_batch_size = meta_batch_size
		self.inner_data = tf.placeholder(tf.float32)
		self.inner_label = tf.placeholder(tf.float32)
		self.outer_data = tf.placeholder(tf.float32)
		self.outer_label = tf.placeholder(tf.float32)
		self.amp = tf.placeholder(tf.float32)
		self.task_loss = []
		for ind in range(self.meta_batch_size):
			if ind == 0:
				task_result = self.meta_learn_task((self.inner_data[ind], self.inner_label[ind], self.outer_data[ind], self.outer_label[ind]), reuse = False)
			else:
				task_result = self.meta_learn_task((self.inner_data[ind], self.inner_label[ind], self.outer_data[ind], self.outer_label[ind]))
			self.task_loss.append(task_result["losses"][-1]/self.amp[ind])
		self.meta_loss = tf.reduce_sum(self.task_loss) / self.meta_batch_size
		self.meta_optimizer = tf.train.AdamOptimizer(self.meta_lr)
		self.meta_gradient = self.meta_optimizer.compute_gradients(self.meta_loss)
		self.meta_train_op = self.meta_optimizer.apply_gradients(self.meta_gradient)
		self.results_path = "./Results"

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		self.saver = tf.train.Saver()

	def create_checkpoint_folders(self, n_iter):
		folder_name = "/{0}_{1}_maml_reg".format(
			str(datetime.datetime.now()).split('.')[0].replace(' ','_'),
			n_iter).replace(':', '-')
		self.model_path = self.results_path + folder_name + "/"
		tensorboard_path = self.results_path + folder_name + '/tensorboard'
		self.saved_model_path = self.results_path + folder_name + '/saved_models/'
		log_path = self.results_path + folder_name + '/log'
		if not os.path.exists(self.results_path + folder_name):
			os.mkdir(self.results_path + folder_name)
			os.mkdir(tensorboard_path)
			os.mkdir(self.saved_model_path)
			os.mkdir(log_path)


	def meta_learn_task(self, input_tensors, scope = "model", reuse = True, test=False):
		inner_data, inner_label, outer_data, outer_label = input_tensors
		task_output = {}
		task_output["losses"] =[]
		task_output["outputs"] = []
		vars_ = self.vars
		if test:
			initial_output = self.forward_prop(outer_data, vars_ , scope = scope,reuse = reuse)
			task_output["outputs"].append(initial_output)
		for ind in range(self.update_per_task):
			if ind == 0:
				inner_output = self.forward_prop(inner_data, vars_ , scope = scope,reuse = reuse)
				inner_loss = self.get_loss(inner_output, inner_label)
				inner_grad = tf.gradients(inner_loss, list(vars_.values()))
				inner_grad_dict = dict(zip(vars_.keys(), inner_grad))
				new_vars = dict(zip(vars_.keys(), [vars_[key] - self.task_lr*inner_grad_dict[key] for key in vars_.keys()]))
			else:
				inner_output = self.forward_prop(inner_data, new_vars, scope = scope)
				inner_loss = self.get_loss(inner_output, inner_label)
				inner_grad = tf.gradients(inner_loss, list(new_vars.values()))
				inner_grad_dict = dict(zip(new_vars.keys(), inner_grad))
				new_vars = dict(zip(new_vars.keys(), [new_vars[key] - self.task_lr*inner_grad_dict[key] for key in new_vars.keys()]))

			outer_output = self.forward_prop(outer_data, new_vars, scope = scope)
			task_output["outputs"].append(outer_output)
			if not test:
				outer_loss = self.get_loss(outer_output, outer_label)
				task_output["losses"].append(outer_loss)
		return task_output

	def load(self, modelpath):
		self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(modelpath))
		return None

	def test_model(self, num_tests = 10):
		dataset = WavedFormData(self.samples_per_task,1, ["triangle"])
		for i in range(num_tests):
			inner_data, inner_label, outer_data, outer_label, amplitude, phase, type_ = dataset.generate_wave_forms_batch()
			inner_data_p = tf.placeholder(tf.float32)
			inner_label_p = tf.placeholder(tf.float32)
			outer_data_p = tf.placeholder(tf.float32)
			outer_label_p = tf.placeholder(tf.float32)
			task = (inner_data_p, inner_label_p, outer_data_p, outer_label_p)
			x = np.arange(-5,5,0.2)
			y = amplitude * signal.square(x - phase)
			task_output = self.sess.run(self.meta_learn_task(task,test=True), feed_dict = {inner_data_p : inner_data[0], inner_label_p : inner_label[0], outer_data_p : x.reshape((-1,1)), outer_label_p :outer_label[0]})

			prediction = task_output["outputs"][-1]
			init_prediction = task_output["outputs"][0]
			fig, ax = plt.subplots()
			ax.plot(x, y, color='#2c3e50', linewidth=0.8, label='Truth')
			ax.scatter(inner_data[0].reshape(-1), inner_label[0].reshape(-1), color='#2c3e50', label='Training Set')
			ax.plot(x, init_prediction.reshape(-1), label='Initial Prediction', color='#0000ff', linestyle=':')
			ax.plot(x, prediction.reshape(-1), label='Prediction', color='#e74c3c', linestyle='--')
			ax.legend()
			plt.savefig("maml-reg_test_multimodal_{}.png".format(i))

	def get_loss(self, output, label):
		return tf.losses.mean_squared_error(label, output)

	def generate_var(self):
		vars_ = {}
		vars_['w1'] = tf.get_variable("w1",shape = [self.input_dim,self.hidden_dims[0]],dtype = tf.float32)
		vars_['b1'] = tf.get_variable("b1", shape = [self.hidden_dims[0]], dtype = tf.float32)
		vars_['w2'] = tf.get_variable("w2",shape = [self.hidden_dims[0],self.hidden_dims[1]],dtype = tf.float32)
		vars_['b2'] = tf.get_variable("b2",shape = [self.hidden_dims[1]],dtype = tf.float32)
		vars_['w3'] = tf.get_variable("w3",shape = [self.hidden_dims[1],self.output_dim],dtype = tf.float32)
		vars_['b3'] = tf.get_variable("b3",shape = [self.output_dim],dtype = tf.float32)
		return vars_

	def forward_prop(self, input_tensor, vars_, scope, reuse = True):
		with tf.variable_scope(scope,reuse = reuse):
			hidden1_out = tf.contrib.layers.layer_norm(tf.matmul(input_tensor, vars_['w1']) + vars_['b1'], activation_fn = tf.nn.relu)
			hidden2_out = tf.contrib.layers.layer_norm(tf.matmul(hidden1_out, vars_['w2']) + vars_['b2'], activation_fn = tf.nn.relu)
			final_out = tf.matmul(hidden2_out , vars_['w3'])  + vars_['b3']
		return final_out

	def train(self,iters = 1000):
		step = 0
		self.create_checkpoint_folders(iters)
		dataset = WavedFormData(self.samples_per_task,self.meta_batch_size)
		for i in range(1, iters + 1):
			print ("----------------------Iter {}/{}-----------------------------".format(i,iters))
			inner_data, inner_label, outer_data, outer_label,amp, phase, type_ = dataset.generate_wave_forms_batch()
			self.sess.run(self.meta_train_op, feed_dict ={self.inner_data :inner_data , self.inner_label :inner_label, self.outer_data :outer_data, self.outer_label :outer_label, self.amp : amplitude})
			# if i%50 == 0:
			loss  = self.sess.run(self.meta_loss,feed_dict ={self.inner_data :inner_data , self.inner_label :inner_label, self.outer_data :outer_data, self.outer_label :outer_label, self.amp :amplitude})
			print("Meta Loss : {}".format(loss))
			step += 1
			self.saver.save(self.sess, save_path=self.saved_model_path, global_step=step)

		return None

if __name__ == "__main__":
	model = MAML_Sinusoid()
	model.train()
	model.test_model()
