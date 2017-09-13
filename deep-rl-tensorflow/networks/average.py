import os
import tensorflow as tf

from .layers import *
from .network import Network
from .cnn import CNN
class AVERAGE_K(Network):
  def __init__(self, sess,
               data_format,
               history_length,
               observation_dims,
               output_size,
               trainable=True,
               hidden_activation_fn=tf.nn.relu,
               output_activation_fn=None,
               weights_initializer=initializers.xavier_initializer(),
               biases_initializer=tf.constant_initializer(0.1),
               value_hidden_sizes=[512],
               advantage_hidden_sizes=[512],
               network_output_type='dueling',
               network_header_type='nips',
               k=1,
               name='AVERAGE'):
    super(AVERAGE_K, self).__init__(sess, name)
    self.k = k #num of networks to average
    self.t = 0 #counter to know how many nets already exist
    self.nets = []

    if data_format == 'NHWC':
      self.inputs = tf.placeholder('float32',
          [None] + observation_dims + [history_length], name='inputs')
    elif data_format == 'NCHW':
      self.inputs = tf.placeholder('float32',
          [None, history_length] + observation_dims, name='inputs')
    else:
      raise ValueError("unknown data_format : %s" % data_format)

    self.var = {}
    self.l0 = tf.div(self.inputs, 255.)
    self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    for i in range(self.k):
        self.nets.append(CNN(sess,
                   data_format,
                   history_length,
                   observation_dims,
                   output_size,
                   trainable,
                   hidden_activation_fn,
                   output_activation_fn,
                   weights_initializer,
                   biases_initializer,
                   value_hidden_sizes,
                   advantage_hidden_sizes,
                   network_output_type,
                   network_header_type,
                   part_of_AVERAGE_K=True,
                   inputs=self.inputs,
                   keep_prob=self.keep_prob,
                   name='target_%d' % i))
    self.build_output_ops()

  def build_output_ops(self):
    ################################### AVERAGE_K build output ops #####################################
    nets_outputs = [net.outputs for net in self.nets]
    self.outputs = tf.reduce_mean(tf.stack(nets_outputs), 0)
    self.max_outputs = tf.reduce_max(self.outputs, reduction_indices=1)
    self.outputs_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
    self.outputs_with_idx = tf.gather_nd(self.outputs, self.outputs_idx)
    self.actions = tf.argmax(self.outputs, dimension=1)
    ################################### AVERAGE_K build output ops  - done #####################################

  def run_copy(self):
    if self.nets[self.t % self.k].copy_op is None:
      raise Exception("run `create_copy_op` for net %d first before copy" % (self.t % self.k))
    else:
      self.sess.run(self.nets[self.t % self.k].copy_op)

  def create_copy_op(self, network):
    for net in self.nets:
        net.create_copy_op(network)
