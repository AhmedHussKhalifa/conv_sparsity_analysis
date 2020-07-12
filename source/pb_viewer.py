import tensorflow as tf

import os 
from tensorflow.python.platform import gfile
MAIN_PATH = '../util'
frozen_graph_path = os.path.join(MAIN_PATH, 'frozen_graphs')
# model_name = 'frozen_pnasnet.pb'

#model_name = 'frozen_vgg_16.pb'
# model_name = 'inception_resnet_v2_frozen.pb'

#model_name = 'frozen_mobilenet_v2_optimized.pb'
model_name = 'classify_image_graph_def.pb'
with tf.Session() as sess:
    # model_filename ='frozen_inception_v1.pb'
    model_filename = os.path.join(frozen_graph_path, model_name)
   
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

LOGDIR= os.path.join(MAIN_PATH, 'frozen_graph_logs/')
print('Output logdir:\n%s' % LOGDIR)
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

train_writer.flush()
train_writer.close()
