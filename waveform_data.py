import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import signal

class WavedFormData(object):
	def __init__(self , n_samples_per_task, batch_size, waveform_list = ["sine", "square", "triangle", "sawtooth"]):
		self.batch_size = batch_size
		type_ = ["sine", "square", "triangle", "sawtooth"]
		self.n_samples_per_task = n_samples_per_task
		self.amp_range = (0.1,5.0)
		self.phase_range = (0,math.pi)
		self.x_range = (-5.0,5.0)
		self.waveform_list = waveform_list
		for waveform in self.waveform_list:
			assert(waveform in type_)

	def generate_wave_forms_batch(self):
		amplitude = np.random.uniform(self.amp_range[0], self.amp_range[1], self.batch_size)
		amplitude = np.ones((self.batch_size,))*5
		phase = np.random.uniform(self.phase_range[0], self.phase_range[1], self.batch_size)
		data_x = np.random.uniform(self.x_range[0], self.x_range[1], [self.batch_size, self.n_samples_per_task*2, 1])
		labels = []
		for task in range(self.batch_size):
			choose_type = random.choice(self.waveform_list)
			if choose_type == "sine":
				task_labels = amplitude[task]*np.sin(data_x[task] - phase[task])
				labels.append(task_labels)
			elif choose_type == "square":
				task_labels = amplitude[task]*signal.square(data_x[task] - phase[task])
				labels.append(task_labels)
			elif choose_type == "triangle":
				task_labels = amplitude[task]*signal.sawtooth(data_x[task] - phase[task], width = 0.5)
				labels.append(task_labels)
			elif choose_type == "sawtooth":
				task_labels = amplitude[task]*signal.sawtooth(data_x[task] - phase[task])
				labels.append(task_labels)
		labels = np.array(labels)
		return data_x[:,:self.n_samples_per_task,:], labels[:,:self.n_samples_per_task,:], data_x[:,self.n_samples_per_task:,:], labels[:,self.n_samples_per_task:,:], amplitude, phase, choose_type
