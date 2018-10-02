import tensorflow as tf
import numpy as np
import datetime
from data_generator import DataGenerator
import os


class MAML_CNN(object):

	def __init__(self,  input_dim = [28,28,1],  meta_batch_size = 32):
		# self.hidden_dims = hidden_dims
		self.input_dim = input_dim
		self.n_classes_per_task = 5
		self.task_lr = 0.4
		self.meta_lr = 0.01
		self.update_per_task = 3
		self.num_shot_train = 5
		self.num_shot_test = 5
		self.vars = self.generate_var()
		self.meta_batch_size = meta_batch_size
		self.data_generator = DataGenerator(datasource = "omniglot", num_classes = self.n_classes_per_task, num_samples_per_class = self.num_shot_train + self.num_shot_test , batch_size =self.meta_batch_size, test_set = False)
		batch_img, batch_label = self.data_generator.make_data_tensor(train=True, load=True, savepath = "omni5w5s.pkl")
		inner_input = tf.slice(batch_img, [0,0,0], [-1,self.n_classes_per_task*self.num_shot_train, -1])
		outer_input = tf.slice(batch_img, [0,self.n_classes_per_task*self.num_shot_train, 0], [-1,-1,-1])
		inner_label = tf.slice(batch_label, [0,0,0], [-1,self.n_classes_per_task*self.num_shot_train, -1])
		outer_label = tf.slice(batch_label, [0,self.n_classes_per_task*self.num_shot_train, 0], [-1,-1,-1])
		inner_input = tf.reshape(inner_input, [-1,self.n_classes_per_task*self.num_shot_train] + input_dim)
		outer_input = tf.reshape(outer_input, [-1,self.n_classes_per_task*self.num_shot_test] + input_dim)
		self.task_loss = []
		for ind in range(self.meta_batch_size):
			if ind == 0:
				task_result = self.meta_learn_task((inner_input[ind], inner_label[ind], outer_input[ind], outer_label[ind]), reuse = False)
			else:
				task_result = self.meta_learn_task((inner_input[ind], inner_label[ind], outer_input[ind], outer_label[ind]))
			self.task_loss.append(task_result["losses"][-1])
		self.meta_loss = tf.reduce_sum(self.task_loss) / self.meta_batch_size
		self.meta_optimizer = tf.train.AdamOptimizer(self.meta_lr)
		self.meta_gradient = self.meta_optimizer.compute_gradients(self.meta_loss)
		self.meta_train_op = self.meta_optimizer.apply_gradients(self.meta_gradient)
		self.results_path = "./Results"
		tf.summary.scalar(name="Model_X_entropy Loss", tensor=self.meta_loss)
		self.summary_op = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		self.saver = tf.train.Saver()

	def create_checkpoint_folders(self, n_iter):
		folder_name = "/{0}_{1}_maml_omni".format(
			str(datetime.datetime.now()).split('.')[0].replace(' ','_'),
			n_iter).replace(':', '-')
		self.model_path = self.results_path + folder_name + "/"
		self.tensorboard_path = self.results_path + folder_name + '/tensorboard'
		self.saved_model_path = self.results_path + folder_name + '/saved_models/'
		if not os.path.exists(self.results_path + folder_name):
			os.mkdir(self.results_path + folder_name)
			os.mkdir(self.tensorboard_path)
			os.mkdir(self.saved_model_path)

	def test_model(self , num_tests = 200):
		data_generator_test = DataGenerator(datasource = "omniglot", num_classes = self.n_classes_per_task, num_samples_per_class = self.num_shot_train + self.num_shot_test , batch_size = 1, test_set = True)
		batch_img, batch_label = data_generator_test.make_data_tensor(train= False)
		inner_input = tf.slice(batch_img, [0,0,0], [-1,self.n_classes_per_task*self.num_shot_train, -1])
		outer_input = tf.slice(batch_img, [0,self.n_classes_per_task*self.num_shot_train, 0], [-1,-1,-1])
		inner_label = tf.slice(batch_label, [0,0,0], [-1,self.n_classes_per_task*self.num_shot_train, -1])
		outer_label = tf.slice(batch_label, [0,self.n_classes_per_task*self.num_shot_train, 0], [-1,-1,-1])
		inner_input = tf.reshape(inner_input, [-1,self.n_classes_per_task*self.num_shot_train] + self.input_dim)
		outer_input = tf.reshape(outer_input, [-1,self.n_classes_per_task*self.num_shot_test] + self.input_dim)
		correct_prediction = 0
		tf.train.start_queue_runners(sess = self.sess)
		for ind in range(num_tests):
			print( "Meta-Testing task {}".format(ind))
			if ind == 0:
				task_result = self.sess.run(self.meta_learn_task((inner_input[0], inner_label[0], outer_input[0], outer_label[0]),reuse = False,  test = True))
			else:
				task_result = self.sess.run(self.meta_learn_task((inner_input[0], inner_label[0], outer_input[0], outer_label[0]), test = True))
			predictions, label = task_result["outputs"][-1]
			predictions = predictions.reshape((self.num_shot_test*self.n_classes_per_task,))
			# print(predictions)
			label = label.reshape((self.num_shot_test*self.n_classes_per_task,))
			# print(label)
			for idx,prediction in enumerate(predictions):
				if prediction == label[idx]:
					correct_prediction += 1
		accuracy = float(correct_prediction/ (num_tests*self.num_shot_test*self.n_classes_per_task))*100
		print("Model Accuracy : {} %".format(accuracy))


	def meta_learn_task(self, input_tensors, scope = "model", reuse = True, test = False):
		inner_data, inner_label, outer_data, outer_label = input_tensors
		task_output = {}
		task_output["losses"] =[]
		task_output["outputs"] = []
		vars_ = self.vars
		for ind in range(self.update_per_task):
			if ind == 0:
				inner_output = self.forward_prop(inner_data, vars_ , scope = scope,reuse = reuse)
				inner_loss = self.get_loss(inner_output, inner_label)
				inner_grad = tf.gradients(inner_loss, list(vars_.values()))
				inner_grad = [tf.stop_gradient(grad) for grad in inner_grad]
				inner_grad_dict = dict(zip(vars_.keys(), inner_grad))
				new_vars = dict(zip(vars_.keys(), [vars_[key] - self.task_lr*inner_grad_dict[key] for key in vars_.keys()]))
			else:
				inner_output = self.forward_prop(inner_data, new_vars, scope = scope)
				inner_loss = self.get_loss(inner_output, inner_label)
				inner_grad = tf.gradients(inner_loss, list(new_vars.values()))
				inner_grad = [tf.stop_gradient(grad) for grad in inner_grad]
				inner_grad_dict = dict(zip(new_vars.keys(), inner_grad))
				new_vars = dict(zip(new_vars.keys(), [new_vars[key] - self.task_lr*inner_grad_dict[key] for key in new_vars.keys()]))
			outer_output = self.forward_prop(outer_data, new_vars, scope = scope)
			if test:
				outer_output = tf.argmax(tf.nn.softmax(outer_output, axis = -1), axis = -1)
				meta_label = tf.argmax(outer_label,axis = -1)
				task_output["outputs"].append((outer_output,meta_label))
			else:
				task_output["outputs"].append(outer_output)
				outer_loss = self.get_loss(outer_output, outer_label)
				task_output["losses"].append(outer_loss)

		return task_output

	def get_loss(self, output, label):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= output,labels = label))

	def generate_var(self):
		vars_ = {}
		for i in range(1,5):
			if i == 1:
				vars_["conv_{}_kernel".format(i)] = tf.get_variable("conv_{}_kernel".format(i), shape = [3,3,self.input_dim[-1], 64], dtype = tf.float32)
			else:
				vars_["conv_{}_kernel".format(i)] = tf.get_variable("conv_{}_kernel".format(i), shape = [3, 3, 64, 64], dtype = tf.float32)
			vars_["conv_{}_bias".format(i)] =tf.get_variable("conv_{}_bias".format(i), shape = [64], dtype = tf.float32)
		vars_["final_dense_weights"] = tf.get_variable("final_dense_weights", shape = [64,self.n_classes_per_task], dtype = tf.float32)
		vars_["final_dense_bias"] = tf.get_variable("final_dense_bias", shape = [self.n_classes_per_task], dtype = tf.float32)
		return vars_

	def forward_prop(self, input_tensor, vars_, scope, reuse = True):
		with tf.variable_scope(scope,reuse = tf.AUTO_REUSE):
			forward_tensor = input_tensor
			for i in range(1,5):
				conv = tf.nn.conv2d(forward_tensor,vars_["conv_{}_kernel".format(i)], [1,1,1,1],'SAME')
				forward_tensor = tf.nn.bias_add(conv,vars_["conv_{}_bias".format(i)])
				forward_tensor = tf.layers.max_pooling2d(forward_tensor,(2,2),(2,2))

				forward_tensor = tf.nn.relu(tf.contrib.layers.batch_norm(forward_tensor, scope = "BatchNorm_{}".format(i)))
			forward_tensor = tf.contrib.layers.flatten(forward_tensor)
			final_out = tf.matmul(forward_tensor, vars_["final_dense_weights"]) + vars_["final_dense_bias"]
		return final_out

	def load(self, modelpath):
		self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(modelpath))
		return None

	def train(self,iters = 2000):
		step = 0
		self.create_checkpoint_folders(iters)
		tf.train.start_queue_runners(sess = self.sess)
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)
		for i in range(1, iters + 1):
			print ("----------------------Iter {}/{}-----------------------------".format(i,iters))
			loss, _  = self.sess.run([self.meta_loss, self.meta_train_op])
			print("Meta Loss : {}".format(loss))
			step += 1
			summary = self.sess.run(self.summary_op)
			self.writer.add_summary(summary, global_step=step)
			self.saver.save(self.sess, save_path=self.saved_model_path, global_step=step)
			step += 1
		return None

if __name__ == "__main__":
	model = MAML_CNN()
	model.train()
