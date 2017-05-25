# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils
#import tflearn

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 4,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L1 regularization."""

  def create_model(self, model_input, vocab_size, l1_penalty=1e-10, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l1_regularizer(l1_penalty))
    output = tf.Print(output, [tf.argmax(output, 1)], 'out = ', summarize = 20, first_n = 10)
    return {"predictions": output}

class PerceptronModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, l1_penalty=1e-10, **unused_params):
        model_input = tf.Print(model_input, [model_input], message = 'model input: ')
        print(model_input.get_shape())
        # input_layer = slim.fully_connected(model_input, 3000, activation_fn=tf.nn.relu) ## .65
        # hidden_layer = slim.fully_connected(input_layer, 3000, activation_fn=tf.nn.relu)
        # output = slim.fully_connected(hidden_layer, vocab_size, activation_fn=tf.nn.softmax)

        # input_layer = slim.fully_connected(model_input, 2000, activation_fn=tf.nn.relu) # training (yt8m_train_video_2000mlp)
        # hidden_layer = slim.fully_connected(input_layer, 2000, activation_fn=tf.nn.relu)
        # output = slim.fully_connected(hidden_layer, vocab_size, activation_fn=tf.nn.softmax)

        # hidden_layer = slim.fully_connected(model_input, 2000, activation_fn=tf.nn.relu) # training (yt8m_train_video_one_layer_2000)
        # output = slim.fully_connected(hidden_layer, vocab_size, activation_fn=tf.nn.softmax)

        hidden_layer_1 = slim.fully_connected(model_input, 2000, activation_fn=tf.nn.relu) #
        hidden_layer_2 = slim.fully_connected(hidden_layer_1, 4000, activation_fn=tf.nn.relu)
        output = slim.fully_connected(hidden_layer_2, vocab_size, activation_fn=tf.nn.softmax)
        output = tf.Print(output, [tf.argmax(output, 1)], 'out = ', summarize = 60, first_n = 100)
        return {"predictions": output}

class ConvModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, l1_penalty=1e-10, **unused_params):
        cnn_input = tf.reshape(model_input, [-1, 1024, 1])
        net = slim.conv2d(cnn_input, 128, [3])
        net = slim.pool(net, [2], "MAX")
        output = slim.fully_connected(net, vocab_size, activation_fn=tf.nn.softmax)
        return {"predictions": output}

class vgg16(models.BaseModel):
    def create_model(self, model_input, vocab_size, l1_penalty=1e-10, **unused_params):
        model_input = tf.Print(model_input, [model_input], message = 'model input: ')
        # print model input using tf.print
        input_layer = tf.reshape(model_input, [-1, 1024, 1])
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(input_layer, 2, slim.conv2d, 64, [3], scope='conv1')
            net = slim.pool(net, [2], "MAX", scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3], scope='conv2')
            net = slim.pool(net, [2], "MAX", scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3], scope='conv3')
            net = slim.pool(net, [2], "MAX", scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3], scope='conv4')
            net = slim.pool(net, [2], "MAX", scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3], scope='conv5')
            net = slim.pool(net, [2], "MAX", scope='pool5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, 0.5, scope='dropout7')
            net = slim.fully_connected(net, vocab_size, activation_fn=None, scope='fc8')
        return {"predictions": net}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

# class NeuralModel(models.BaseModel):
#     # a neural network model
#   def create_model(self, model_input, vocab_size, weights, biases, **unused_params):
#     print model_input.get_shape()[1]
#     print vocab_size
#     # Hidden layer with RELU activation
#     layer_1 = tf.add(tf.matmul(model_input, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     # Hidden layer with RELU activation
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     # Output layer with linear activation
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     print "OUT LAYER"
#     print out_layer
#     return {"predictions": out_layer}