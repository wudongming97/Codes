import tensorflow as tf
from utils import convert2float


def _info(name):
    return {
        'apple': ('datas/apple2orange/trainA', 'datas/apple.tfrecords'),
        'orange': ('datas/apple2orange/trainB', 'datas/orange.tfrecords'),
        'horse': ('datas/horse2zebra/trainA', 'datas/horse.tfrecords'),
        'zebra': ('datas/horse2zebra/trainB', 'datas/zebra.tfrecords')
    }[name]


class Reader():
    def __init__(self, name, image_size=256,
                 min_queue_examples=1000, num_threads=4):
        self.name = name
        self.tfrecords_file = _info(self.name)[1]
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.filename_queue = tf.train.string_input_producer([self.tfrecords_file])

    def feed(self, batch_size=1):
        """
        Returns:
            images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            _, serialized_example = self.reader.read(self.filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string)
                })

            image_buffer = features['image/encoded_image']
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = self._preprocess(image)

            images = tf.train.shuffle_batch(
                [image], batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples * 5, min_after_dequeue=self.min_queue_examples
            )
        return images

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = convert2float(image)
        image.set_shape([self.image_size, self.image_size, 3])
        return image


def test_reader():
    with tf.Graph().as_default():
        reader1 = Reader('apple')
        reader2 = Reader('orange')
        images_op1 = reader1.feed()
        images_op2 = reader2.feed()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                batch_images1, batch_images2 = sess.run([images_op1, images_op2])
                print("image shape: {}".format(batch_images1))
                print("image shape: {}".format(batch_images2))
                print("=" * 10)
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
