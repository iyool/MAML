"""
Code for generating and loading data
"""

import numpy as np
import os
import random
import tensorflow as tf
import pickle
from absl import flags
from absl import app

from utils import get_images


FLAGS = flags.FLAGS

## Copied from Finn's implementation https://github.com/cbfinn/maml/blob/master/data_generator.py
## code borrowed with permission from @greentfrapp


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, datasource, num_classes, num_samples_per_class, batch_size, test_set, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.datasource = datasource
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size
        self.test_set = test_set

        if self.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif self.datasource == 'omniglot':
            self.num_classes = config.get('num_classes', self.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if self.test_set:
                self.metaval_character_folders = character_folders[num_train+num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif self.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', self.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/train')
            if self.test_set:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        elif self.datasource == 'cifar':
            self.num_classes = config.get('num_classes', self.num_classes)
            self.img_size = config.get('img_size', (32, 32))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/cifar/train')
            if self.test_set:
                metaval_folder = config.get('metaval_folder', './data/cifar/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/cifar/val')

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')


    def make_data_tensor(self, train=True, save=False, load=False, savepath=None):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        if self.datasource in ['omniglot', 'cifar', 'miniimagenet']:
            if load:
                assert train, 'Loading only allowed for training set'
                assert savepath is not None, 'Savepath must be set'
                with open(savepath, 'rb') as file:
                    all_filenames = pickle.load(file)
                labels = list(np.concatenate(np.array(self.num_samples_per_class * [list(range(self.num_classes))]).T, axis=0))
                print('Loaded training set from {}'.format(savepath))
            else:
                print('Generating filenames')
                if save:
                    assert train, 'Saving only allowed for training set'
                    assert savepath is not None, 'Savepath must be set'
                all_filenames = []
                from datetime import datetime
                start = datetime.now()
                for i in range(num_total_batches):
                    if (i + 1) % 5000 == 0:
                        print('Generated {}/{} tasks...'.format((i + 1), num_total_batches))
                    sampled_character_folders = random.sample(folders, self.num_classes)
                    random.shuffle(sampled_character_folders)

                    labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                    # make sure the above isn't randomized order
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                if save:
                    with open(savepath, 'wb') as file:
                        pickle.dump(all_filenames, file)
                    print('\nTraining set saved to {}'.format(savepath))
                    print('\nDetails:')
                    print('--datasource={}'.format(FLAGS.datasource))
                    print('--num_classes={}'.format(FLAGS.num_classes))
                    print('--num_shot_train={}'.format(FLAGS.num_shot_train))
                    print('--num_shot_test={}'.format(FLAGS.num_shot_test))
                    print('--savepath={}'.format(FLAGS.savepath))
                    quit()

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if self.datasource in ['miniimagenet', 'cifar']:
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0],self.img_size[1],3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0],self.img_size[1],1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if self.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                if self.datasource == 'omniglot': # and FLAGS.train:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
                        k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing - the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase

def main(unused_args):
    if FLAGS.save:
        assert FLAGS.datasource in ['omniglot', 'cifar', 'miniimagenet'], '--datasource should be one of [\'omniglot\', \'cifar\', \'miniimagenet\']'
        assert FLAGS.savepath is not None, 'Use --savepath to set where to save the training set'

        data_generator = DataGenerator(
            datasource=FLAGS.datasource,
            num_classes=FLAGS.num_classes,
            num_samples_per_class=FLAGS.num_shot_train+FLAGS.num_shot_test,
            batch_size=1,
            test_set=False,
        )
        data_generator.make_data_tensor(train=True, save=True, savepath=FLAGS.savepath)

    else:
        print('Run the script with the --save flag to save a training set')

if __name__ == '__main__':

    # Defining options for saving the training set, since it can take up to 30 minutes to generate, especially for cifar and miniimagenet

    flags.DEFINE_bool('save', False, 'Save the training set')
    flags.DEFINE_string('datasource', 'omniglot', 'Should be one of [\'omniglot\', \'cifar\', \'miniimagenet\']')
    flags.DEFINE_integer('num_classes', 5, 'Number of classes/ways')
    flags.DEFINE_integer('num_shot_train', 1, 'Number of training samples per class')
    flags.DEFINE_integer('num_shot_test', 1, 'Number of test samples per class')
    flags.DEFINE_string('savepath', None, 'Path to save training set to')

    app.run(main)


