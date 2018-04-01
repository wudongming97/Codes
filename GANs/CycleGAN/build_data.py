# from https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/build_data.py
import os

import tensorflow as tf

from data import _info


def file_paths(dir):
    paths = []
    for img in os.scandir(dir):
        if img.name.endswith('.jpg'):
            paths.append(img.path)
    return paths


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
    """Build an Example proto for an example.
    Args:
      file_path: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
    Returns:
      Example proto
    """
    file_name = file_path.split('/')[-1]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
        'image/encoded_image': _bytes_feature(image_buffer)
    }))
    return example


def data_writer(dir, output_file):
    """Write data to tfrecords
    """
    _paths = file_paths(dir)
    _num = len(_paths)

    # dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(len(_paths)):
        file_path = _paths[i]

        with tf.gfile.FastGFile(file_path, 'rb') as f:
            image_data = f.read()

        example = _convert_to_example(file_path, image_data)
        writer.write(example.SerializeToString())

        if i % 500 == 0:
            print("Processed {}/{}.".format(i, _num))
    print("Done.")
    writer.close()


if __name__ == '__main__':
    datasets = ['apple', 'orange', 'horse', 'zebra']
    for ds in datasets:
        data_info = _info(ds)
        data_writer(data_info[0], data_info[1])
