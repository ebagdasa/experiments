import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector
LOG_DIR = '/home/ubuntu/projects/tf.log/'

# Create randomly initialized embedding weights which will be trained.
N = 24541 # Number of items (vocab size).
D = 200 # Dimensionality of the embedding.

def get_values():
    npa = []
    with open('/home/ubuntu/projects/obj/edges_small_test.output') as f:
        lines = f.readlines()
        dicarr = dict()
        for line in lines[1:]:
            values = line.split(' ')
            dicarr[int(values[0])] = [float(x) for x in values[1:]]
        npa = np.array([dicarr[y] for y in sorted(list(dicarr.keys()))])
    np.save('/home/ubuntu/projects/obj/nparray_small.npy', npa)

    return npa


npa = get_values()
#np.load('/home/ubuntu/projects/mycode/snap/examples/node2vec/nparray.npy')
embedding_var = tf.Variable(npa, name='word_embedding')

# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'github.tsv')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)
# summary_writer

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 0)