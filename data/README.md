# Few-shot Datasets

Paths described below are relative to the current directory `data` eg. `omniglot_resized` refers to `data/omniglot_resized/`.

## Omniglot

First download the omniglot dataset from https://github.com/brendenlake/omniglot and extract the contents of both `images_background` and `images_evaluation` into `omniglot_resized` so you should have paths like `omniglot_resized/Alphabet_of_the_Magi/character01/0709_01.png`.

Then, run the following:

```
$ cd omniglot_resized/
$ python resize_images.py
```

This resizes the images to 28 by 28 pixels.

## CIFAR-FS

First download the CIFAR-100 dataset from http://www.cs.toronto.edu/~kriz/cifar.html

***Download CIFAR-100 and NOT CIFAR-10***

Extract the contents of cifar-100-python to `cifar`.

Then run the following:

```
$ cd cifar/
$ python proc_images.py
```

This transposes the 32 by 32 pixel images to be channel-last and places them in the `test`, `train` and `val` folders.

## MiniImageNet

I can't seem to find a good source for this dataset so the simplest way to do this is to just download the dataset from my Google Drive [here](https://drive.google.com/file/d/16pifyDIvxxI0ILEtw587-Kpx1HcaU9e3/view?usp=sharing).

Then just extract the entire `miniImagenet` folder into the current folder (`data`).
This is already resized and split into `test`, `val` and `train` folders.

## Additional Notes

When running experiments, the `make_data_tensor` method in `data_generator.py` actually generates 2e+5 training tasks by randomly selecting combinations of classes and samples to make N-way k-shot tasks. This can take a while, possibly more than 30 minutes for CIFAR-FS and MiniImageNet. 

An alternative is to generate a set of 2e+5 training tasks once, save the list of filenames as a pickle file and then load from this file when running experiments. This is significantly faster, taking less than a minute to load after the first time. 

A separate pickle file has to be generated for each type of task (5way1shot vs 5way5shot) and for each datasource.

**Once saved, be careful to load the correct pickle file ie. way and shot used in the experiment must match the loaded file.** This is a potential source of a hidden bug, which will cause the model to fail to learn anything.

The training set can be saved easily by calling the `data_generator.py` script and specifying the `--save`  and other task-specific flags:

```
$ python data_generator.py --save --savepath='my_training_set.pkl' --datasource='cifar' --num_classes=5 --num_shot_train=1 --num_shot_test=1
```

Subsequently, when calling the `make_data_tensor` method with an initialized `DataGenerator` object, just specify the arguments `load=True` and `savepath='my_training_set.pkl'`.

This is only enabled for the training set, since the validation and test sets are significantly faster to generate (<< 1 minute).

## Instructions for using `DataGenerator`

The `DataGenerator` class is used to generate the sinusoid toy task, as well as the few-shot classification tasks from above.

Generating the sinusoid toy task is straightforward, just initialize a `DataGenerator` object then generate samples and labels and pass these to any `tf.placeholder` variables using `sess.run`.

Example:

```
data_generator = DataGenerator(
	datasource=sinusoid,
	num_classes=None,
	num_samples_per_class=num_shot_train+num_shot_test,
	batch_size=meta_batch_size,
	test_set=None,
)
batch_x, batch_y, amp, phase = data_generator.generate()
train_inputs = batch_x[:, :num_shot_train, :]
train_labels = batch_y[:, :num_shot_train, :]
test_inputs = batch_x[:, num_shot_train:, :]
test_labels = batch_y[:, num_shot_train:, :]
feed_dict = {
	model.train_inputs: train_inputs,
	model.train_labels: train_labels,
	model.test_inputs: test_inputs,
	model.test_labels: test_labels,
	model.amp: amp, # use amplitude to scale loss
}
loss, _ = sess.run([model.loss, model.optimize], feed_dict)
```

Generating the classification tasks is different, since the classification tasks use a Tensorflow queue to efficiently pass data. 

Again, we first initialize a `DataGenerator` object then generate samples and labels. But these generated samples and labels are Tensorflow tensors, which do not have any value until we run `tf.train.start_queue_runners()`. Instead, we pass these directly to any network layers, without the need for `tf.placeholder` objects.

Example:

```
data_generator = DataGenerator(
	datasource=omniglot,
	num_classes=num_classes,
	num_samples_per_class=num_shot_train+num_shot_test,
	batch_size=meta_batch_size,
	test_set=False,
)

# assuming we saved the dataset in test.pkl and we want to load from it
# otherwise, set load=False and savepath=None
image_tensor, label_tensor = data_generator.make_data_tensor(train=True, load=True, savepath='test.pkl') 

# here we slice the tensors into training and test
train_inputs = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes*num_shot_train, -1])
test_inputs = tf.slice(image_tensor, [0, num_classes*num_shot_train, 0], [-1, -1, -1])
train_labels = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes*num_shot_train, -1])
test_labels = tf.slice(label_tensor, [0, num_classes*num_shot_train, 0], [-1, -1, -1])

input_tensors = {
	'train_inputs': train_inputs, # Shape is (batch_size, num_classes * num_shot_train, 28 * 28)
	'train_labels': train_labels, # Shape is (batch_size, num_classes * num_shot_train, num_classes)
	'test_inputs': test_inputs, # Shape is (batch_size, num_classes * num_shot_test, 28 * 28
	'test_labels': test_labels, # Shape is (batch_size, num_classes * num_shot_test, num_classes)
}

# initialize a model with the input tensors eg.

conv_1 = tf.layers.conv2d(
	inputs=tf.reshape(input_tensors['train_inputs'], [-1, 28, 28, 1]),
	filters=64,
	kernel_size=(3, 3),
	strides=(1, 1),
)

# calculate loss with input tensors

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(input_tensors['train_labels'], [-1, num_classes]), logits=self.logits))
```

I found it easier to initialize separate models for training and validation and then using `reuse=tf.AUTO_REUSE` to share parameters.

Also note that for validation, set `test_set=False` when initializing the `DataGenerator` and pass `train=False` as an argument when calling `make_data_tensor`.

For testing, set `test_set=False` when initializing the `DataGenerator` and pass `train=False` when calling `make_data_tensor`.
