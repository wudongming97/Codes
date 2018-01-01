import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('steps', 200000, '')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('lr', 0.0005, '')
flags.DEFINE_float('beta', 1.0, '')
flags.DEFINE_float('alpha', 0.0, 'the frac of aux_loss')
flags.DEFINE_float('gamma', 5.0, '')

# ================== VAE_katong =======================
flags.DEFINE_string('model_name', 'VAE_cifa10', '')
flags.DEFINE_string('file_name', './data/cifar10_0.tfrecords', '')
flags.DEFINE_string('ckpt_path', './results/VAE_cifa10/ckpt/', '')
flags.DEFINE_string('logs_path', './results/VAE_cifa10/logs/', '')
# ================= VAE_katong ========================
FLAGS = flags.FLAGS
SHAPE = [32, 32, 3]