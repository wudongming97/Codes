import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('z_size', 16, '')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('beta', 0.1, '')

# ================== VAE_mnist =======================
flags.DEFINE_string('model_name', 'VAE_mnist', '')
flags.DEFINE_string('file_name', './mnist.tfrecords', '')
flags.DEFINE_string('ckpt_path', './results/VAE_mnist/ckpt/', '')
flags.DEFINE_string('logs_path', './results/VAE_mnist/logs/', '')
# ================= VAE_mnist ========================
FLAGS = flags.FLAGS
SHAPE = [28, 28, 1]