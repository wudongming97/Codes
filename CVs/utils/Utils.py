import os

import tensorflow as tf
from matplotlib.image import imread


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 获取目录下所有以suffix为后缀的文件名
def get_file_name_by_suffix(dir_, suffix):
    return [os.path.join(dir_, name) for name in os.listdir(dir_)
            if os.path.isfile(os.path.join(dir_, name)) and name.endswith(suffix)]


def get_file_name(dir_):
    return [os.path.join(dir_, name) for name in os.listdir(dir_)
            if os.path.isfile(os.path.join(dir_, name))]


def fixed_image_writer(images_path, file_name):
    writer = tf.python_io.TFRecordWriter(file_name)
    for path_ in get_file_name(images_path):
        image = imread(path_)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image.tostring())
        }))
        writer.write(example.SerializeToString())
    writer.close()


def fixed_image_reader(filename_list, shape_=None):
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


if __name__ == '__main__':
    fixed_image_writer('./raw', 'cifar10.tfrecords')
