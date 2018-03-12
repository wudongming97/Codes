import os
import pickle

import numpy as np
import tensorflow as tf
from matplotlib.image import imread


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 获取目录下所有以suffix为后缀的文件名
def get_file_name_by_suffix(dir_, suffix_):
    return [os.path.join(dir_, name) for name in os.listdir(dir_)
            if os.path.isfile(os.path.join(dir_, name)) and name.endswith(suffix_)]


def get_file_name_by_prefix(dir_, prefix_):
    return [os.path.join(dir_, name) for name in os.listdir(dir_)
            if os.path.isfile(os.path.join(dir_, name)) and name.startswith(prefix_)]


def get_file_name(dir_):
    return [os.path.join(dir_, name) for name in os.listdir(dir_)
            if os.path.isfile(os.path.join(dir_, name))]


def single_image_writer(writer, image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image.tostring())
    }))
    writer.write(example.SerializeToString())


def fixed_image_writer(filename_list, target_file):
    writer = tf.python_io.TFRecordWriter(target_file)
    for path_ in filename_list:
        image = imread(path_)
        single_image_writer(writer, image)
    writer.close()


def fixed_image_reader(filename_list, shape_):
    queue_ = tf.train.string_input_producer(filename_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue_)
    # 解析
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255)
    # 将图片转换成多维数组形式
    image = tf.reshape(image, shape_)
    image.set_shape(shape_)
    return image


def train_is_ok(sess, steps):
    return sess.run(tf.train.get_global_step()) >= steps


def print_shape(v):
    print(v.get_shape().as_list())


def cifar10_unpickle(file):
    with open(file, 'rb') as fo:
        dick = pickle.load(fo, encoding='bytes')
    return dick


def cifar10_tfrecord_writer(dir_, target, cls=None):
    from os.path import join
    writer = tf.python_io.TFRecordWriter(target)
    data_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    # label_names = cifar10_unpickle(join(dir_, 'batches.meta'))[b'label_names']
    files_ = [join(dir_, fi) for fi in data_names]
    for file in files_:
        dick = cifar10_unpickle(file)
        data, lables, filenames = dick[b'data'], dick[b'labels'], dick[b'filenames']
        for ix, l_ in enumerate(lables):
            image = np.reshape(np.reshape(data[ix], [3, 1024]).T, [32, 32, 3])
            if cls is None:
                single_image_writer(writer, image)
            elif l_ in cls:
                single_image_writer(writer, image)
            else:
                None

    writer.close()

def mnist_tfrecord_writer(dir_, target):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(dir_, dtype=tf.uint8)
    writer = tf.python_io.TFRecordWriter(target)
    for i in range(mnist.train.num_examples):
        image = mnist.train.images[i]
        single_image_writer(writer, image)
    writer.close()


if __name__ == '__main__':
    # filename_list = get_file_name_by_prefix('./raw', '0_')
    # fixed_image_writer(filename_list, 'cifar10_0.tfrecords')

    # cifar10_tfrecord_writer('E:\\data\\cifar-10-batches-py\\', 'cifar10_0.tfrecords', cls=[0])
    mnist_tfrecord_writer('/home/yx/Datasets/MNIST/', 'mnist.tfrecords')
