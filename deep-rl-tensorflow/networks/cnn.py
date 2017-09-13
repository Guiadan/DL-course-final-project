import os
import tensorflow as tf

from .layers import *
from .network import Network

class CNN(Network):
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
               dropout_keep_probability=0.5,
               part_of_AVERAGE_K=False,
               inputs=None,
               keep_prob=None,
               name='CNN'):
    super(CNN, self).__init__(sess, name)

    if part_of_AVERAGE_K:
        if inputs == None:
            raise ValueError("inputs layer not defined")
        else:
            self.inputs = inputs

        if keep_prob == None:
            raise ValueError("keep_prob op is not defined")
        else:
            self.keep_prob = keep_prob
    else:
        if data_format == 'NHWC':
          self.inputs = tf.placeholder('float32',
              [None] + observation_dims + [history_length], name='inputs')
        elif data_format == 'NCHW':
          self.inputs = tf.placeholder('float32',
              [None, history_length] + observation_dims, name='inputs')
        else:
          raise ValueError("unknown data_format : %s" % data_format)
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    self.var = {}
    self.l0 = tf.div(self.inputs, 255.)

    with tf.variable_scope(name):
        if network_header_type.lower() == 'nature':
            self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
                32, [8, 8], [4, 4], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l1_conv')
            self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
                64, [4, 4], [2, 2], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l2_conv')
            self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2,
                64, [3, 3], [1, 1], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l3_conv')
            self.drop = tf.nn.dropout(self.l3, self.keep_prob)
            self.l4, self.var['l4_w'], self.var['l4_b'] = \
                linear(self.drop, 512, weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l4_conv')
            layer = self.l4
        elif network_header_type.lower() == 'nips':
            self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
                16, [8, 8], [4, 4], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l1_conv')
            self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
                32, [4, 4], [2, 2], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l2_conv')
            self.drop = tf.nn.dropout(self.l2, self.keep_prob)
            self.l3, self.var['l3_w'], self.var['l3_b'] = \
                linear(self.drop, 256, weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l3_conv')
            layer = self.l3
        else:
            raise ValueError('Wrong DQN type: %s' % network_header_type)

        self.build_output_ops(layer, network_output_type,
              value_hidden_sizes, advantage_hidden_sizes, output_size,
              weights_initializer, biases_initializer, hidden_activation_fn,
              output_activation_fn, trainable)
