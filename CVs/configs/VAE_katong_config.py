import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('steps', 200000, '')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float('lr', 0.0005, '')
flags.DEFINE_float('beta', 1.0, '')

# ================== VAE_katong =======================
flags.DEFINE_string('model_name', 'VAE_katong', '')
flags.DEFINE_string('file_name', './data/katong.tfrecords', '')  # 16383 * [96, 96, 3]
flags.DEFINE_string('ckpt_path', './results/VAE_katong/ckpt/', '')
flags.DEFINE_string('logs_path', './results/VAE_katong/logs/', '')
# ================= VAE_katong ========================
FLAGS = flags.FLAGS
SHAPE = [96, 96, 3]