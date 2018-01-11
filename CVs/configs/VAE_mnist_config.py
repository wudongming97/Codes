import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('beta', 1.0, '')
flags.DEFINE_float('alpha', 1.0, 'the frac of aux_loss')
flags.DEFINE_float('gamma', 2.0, '')

# ================== VAE_mnist =======================
flags.DEFINE_string('model_name', 'VAE_mnist', '')
flags.DEFINE_string('file_name', './data/mnist.tfrecords', '')
flags.DEFINE_string('ckpt_path', './results/VAE_mnist/ckpt/', '')
flags.DEFINE_string('logs_path', './results/VAE_mnist/logs/', '')
# ================= VAE_mnist ========================
FLAGS = flags.FLAGS
SHAPE = [28, 28, 1]