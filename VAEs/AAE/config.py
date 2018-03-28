import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('steps', 6000, '')
flags.DEFINE_integer('bz', 64, '')
flags.DEFINE_integer('z_dim', 16, '')
flags.DEFINE_float('alpha', 10, 'wgan 惩罚正则项的权重')

flags.DEFINE_string('log_path', './results/', '')

FLAGS = flags.FLAGS
