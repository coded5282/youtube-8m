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
import model_utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
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

        # input_layer = slim.fully_connected(model_input, 6000, activation_fn=tf.nn.relu) # yt8m_train_skip_mlp1
        # drop_layer_1 = slim.dropout(input_layer, 0.5)
        # hidden_layer_1 = slim.fully_connected(drop_layer_1, 6000, activation_fn=tf.nn.relu)
        # drop_layer_2 = slim.dropout(hidden_layer_1, 0.5)
        # skip_layer = tf.add(input_layer, drop_layer_2)
        # activate_layer = tf.nn.relu(skip_layer)
        # output = slim.fully_connected(activate_layer, vocab_size, activation_fn=tf.nn.softmax)

        input_layer = slim.fully_connected(model_input, 4000, activation_fn=tf.nn.relu) #
        drop_layer_1 = slim.dropout(input_layer, 0.3)
        hidden_layer_1 = slim.fully_connected(drop_layer_1, 4000, activation_fn=tf.nn.relu)
        drop_layer_2 = slim.dropout(hidden_layer_1, 0.3)
        skip_layer = tf.add(input_layer, drop_layer_2)
        activate_layer = tf.nn.relu(skip_layer)
        output = slim.fully_connected(activate_layer, vocab_size, activation_fn=tf.nn.softmax)

        output = tf.Print(output, [tf.argmax(output, 1)], 'out = ', summarize = 60, first_n = 100)
        return {"predictions": output}

class ResModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, is_training, l1_penalty=1e-10, **unused_params):

        if is_training == True:
            drop_prob = 0.5
        else:
            drop_prob = 1

        input_layer = slim.fully_connected(model_input, 10000, activation_fn=tf.nn.relu)
        drop_layer_1 = slim.dropout(input_layer, drop_prob)
        hidden_layer = slim.fully_connected(drop_layer_1, 10000, activation_fn=tf.nn.relu)
        drop_layer_2 = slim.dropout(hidden_layer, drop_prob)
        skip_layer = tf.add(input_layer, drop_layer_2)
        output = slim.fully_connected(skip_layer, vocab_size, activation_fn=tf.nn.softmax)
        return {"predictions": output}
        pass

class AutoModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, l1_penalty=1e-10, **unused_params):
        input_layer = slim.fully_connected(model_input, 1152, activation_fn=tf.nn.relu)
        hidden_layer = slim.fully_connected(input_layer, 300, activation_fn=tf.nn.relu)
        output = slim.fully_connected(hidden_layer, vocab_size, activation_fn=tf.nn.softmax)
        return {"predictions": output}


class ConvModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, l1_penalty=1e-10, **unused_params):
        # cnn_input = tf.reshape(model_input, [-1, 1024, 1])
        # net = slim.conv2d(cnn_input, 32, [1])
        # net = slim.pool(net, [1], "MAX")
        # net = slim.flatten(net)
        # output = slim.fully_connected(net, vocab_size, activation_fn=tf.nn.softmax)

        cnn_input = tf.reshape(model_input, [-1, 1152, 1])
        net = slim.conv2d(cnn_input, 32, [1])
        net = slim.pool(net, [1], "MAX")
        net = slim.flatten(net)
        net = slim.fully_connected(net, 6000, activation_fn=tf.nn.relu)
        net = slim.dropout(net, 0.5)
        output = slim.fully_connected(net, vocab_size, activation_fn=tf.nn.softmax)
        return {"predictions": output}

class vgg16(models.BaseModel):
    def create_model(self, model_input, vocab_size, l1_penalty=1e-10, **unused_params):
        model_input = tf.Print(model_input, [model_input], message = 'model input: ')
        # print model input using tf.print
        input_layer = tf.reshape(model_input, [-1, 1152, 1])
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
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, 0.5, scope='dropout7')
            net = slim.fully_connected(net, vocab_size, activation_fn=tf.nn.softmax, scope='fc8')
        return {"predictions": net}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   prefix='',
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
##################################################################################################################################
class ComplexMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   prefix='',
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
    expert_mid_activations = slim.fully_connected(
        model_input,
        2048,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="expertsMid")
    expert_activations = slim.fully_connected(
        expert_mid_activations,
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
#############################################################################################################################
class MLPModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a MLP model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = model_utils.make_fully_connected_net(model_input,
        [512, 256], vocab_size, l2_penalty)
    return {"predictions": output}

class MLPModelV2(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fully_connected_net(model_input,
          [784, 512, 256], vocab_size, l2_penalty)
      return {"predictions": output}

class MLPModel384(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fully_connected_net(model_input,
          [784, 512, 384], vocab_size, l2_penalty)
      return {"predictions": output}

class DeepMLPModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fully_connected_net(model_input,
          [784, 512, 512, 512, 256], vocab_size, l2_penalty)
      return {"predictions": output}

class BatchNormMLP(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fully_connected_net(model_input,
          [784, 512, 512, 512, 256], vocab_size, l2_penalty, batch_norm=True)
      return {"predictions": output}

class SkipConnections(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fcnet_with_skips(model_input,
          [784, 512, 512, 512, 256], [(0, 3), (2, 4)], vocab_size, l2_penalty)
      return {"predictions": output}

class DeepSkip(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fcnet_with_skips(model_input,
          [784] + [512]*8,
          [(0, 3), (2, 4), (4, 6), (6, 8)], vocab_size, l2_penalty)
      return {"predictions": output}

class BigNN(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fcnet_with_skips(model_input,
          [1024] + [768]*8,
          [(0, 3), (2, 4), (4, 6), (6, 8)], vocab_size, l2_penalty)
      return {"predictions": output}

class BiggerNN(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fcnet_with_skips(model_input,
          [1536] + [1024]*8,
          [(0, 3), (2, 4), (4, 6), (6, 8)], vocab_size, l2_penalty)
      return {"predictions": output}

class DeeperSkip(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fcnet_with_skips(model_input,
          [784] + [512]*14,
          [(0, 3), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)],
          vocab_size, l2_penalty)
      return {"predictions": output}

class MidSkip(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fcnet_with_skips(model_input,
            [1536, 1024],
            [(0, 2)],
            vocab_size, l2_penalty)
      return {"predictions": output}

class SmallSkip(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
      output = model_utils.make_fcnet_with_skips(model_input,
            [1536],
            [(0, 2)],
            vocab_size, l2_penalty)
      return {"predictions": output}
#######################################################################################################
class Model3nn2048BnReluSkipDouble(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_hidden_units=2048, l2_penalty=1e-8, prefix='', **unused_params):
      """Creates a logistic model.
      Args:
        model_input: 'batch' x 'num_features' matrix of input features.
        vocab_size: The number of classes in the dataset.
      Returns:
        A dictionary with a tensor containing the probability predictions of the
        model in the 'predictions' key. The dimensions of the tensor are
        batch_size x num_classes."""
      # Initialize weights for projection
      w_s = tf.Variable(tf.random_normal(shape=[1152, 2048], stddev=0.01))
      input_projected = tf.matmul(model_input, w_s)

      hidden1 = tf.layers.dense(
          inputs=model_input, units=num_hidden_units, activation=None,
          kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix + 'fc_1')

      bn1 = slim.batch_norm(hidden1, scope=prefix + 'bn1')

      relu1 = tf.nn.relu(bn1, name=prefix + 'relu1')

      input_projected_plus_h1 = tf.add(input_projected, relu1)

      hidden2 = tf.layers.dense(
          inputs=input_projected_plus_h1, units=num_hidden_units, activation=None,
          kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix + 'fc_2')

      bn2 = slim.batch_norm(hidden2, scope=prefix + 'bn2')

      relu2 = tf.nn.relu(bn2, name=prefix + 'relu2')

      input_projected_plus_h2 = tf.add_n([input_projected, input_projected_plus_h1, relu2])
      # input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

      output = slim.fully_connected(
          input_projected_plus_h2, vocab_size, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix + 'fc_3')

      weights_norm = tf.add_n(tf.losses.get_regularization_losses())

      return {"predictions": output, "regularization_loss": weights_norm}
      # return {"predictions": output}
##############################################################################################################################
class TwoLayerModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_hidden_units=2048, l2_penalty=1e-8, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    hidden1 = slim.fully_connected(
        model_input, num_hidden_units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_1')

    hidden1 = slim.dropout(hidden1, 0.5, scope=prefix+"dropout1")

    output = slim.fully_connected(
        hidden1, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_2')

    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm,"hidden_features": hidden1}
    #return {"predictions": output}

class NeuralAverageModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_2048a = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048a/')
    output_2048b = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048b/')
    output_2048c = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048c/')
    output_2048d = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048d/')

    t1 = output_2048a["predictions"]
    t2 = output_2048b["predictions"]
    t3 = output_2048c["predictions"]
    t4 = output_2048d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])

    scalar = tf.constant(0.25)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class StackModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a Stack of Neural Networks Model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_2048a = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048a/')
    output_2048b = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048b/')
    output_2048c = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048c/')
    output_2048d = TwoLayerModel().create_model(model_input, vocab_size,\
                               num_hidden_units=2048, l2_penalty=l2_penalty, prefix='u2048d/')

    t1 = output_2048a["hidden_features"]
    t2 = output_2048b["hidden_features"]
    t3 = output_2048c["hidden_features"]
    t4 = output_2048d["hidden_features"]

    stacked_features = tf.concat([t1, t2, t3, t4], 1)
    stacked_fc1 = slim.fully_connected(
      stacked_features,
      2048,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="Stack/fc1")
    stacked_fc1 = slim.dropout(stacked_fc1, 0.5, scope="Stack/dropout1")
    stacked_fc2 = slim.fully_connected(
      stacked_fc1,
      vocab_size,
      activation_fn=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="Stack/fc2")

    output = tf.nn.sigmoid(stacked_fc2)

    #return {"predictions": output, "regularization_loss": weights_norm}
    return {"predictions": output}
##############################################################################################################################
class FCModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, nb_units=2000, **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(model_input, nb_units, scope="fc1")
    output = slim.dropout(output, 0.5, scope="dropout1")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc2")
    return {"predictions": output}

class FC2Model(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, nb_units=1000, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(model_input, nb_units, scope="fc1",
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.dropout(output, 0.5, scope="dropout1")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc2",
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class FC3Model(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, nb_units=2000, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(model_input, nb_units, scope="fc1",
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.dropout(output, 0.5, scope="dropout1")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc2",
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class FC4Model(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, nb_units=2000, l2_penalty=1e-8, **unused_params):
    output = slim.fully_connected(model_input, nb_units, scope="fc1",
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(output, nb_units, scope="fc2",
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.dropout(output, 0.5, scope="dropout1")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc3",
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class FC5Model(models.BaseModel):

  def create_model(self, model_input, vocab_size, nb_units=2000, l2_penalty=1e-8, **unused_params):
    output = slim.fully_connected(model_input, nb_units, scope="fc1",
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.dropout(output, 0.8, scope="dropout1")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc2",
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class FC6Model(models.BaseModel):

  def create_model(self, model_input, vocab_size, nb_units=2000, l2_penalty=1e-7, **unused_params):
    output = slim.fully_connected(model_input, nb_units, scope="fc1",
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.dropout(output, 0.1, scope="dropout1")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc2",
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class FC7Model(models.BaseModel):
  def create_model(self, model_input, vocab_size, nb_units=3000, l2_penalty=1e-8, **unused_params):
    output = slim.fully_connected(model_input, nb_units, scope="fc1",
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.dropout(output, 0.1, scope="dropout1")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc2",
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class FCBNModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, nb_units=2000, **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(model_input, nb_units, scope="fc1")
    output = slim.batch_norm(output, scope="bn1")
    output = slim.dropout(output, 0.5, scope="dropout1")
    output = slim.fully_connected(output, nb_units, scope="fc2")
    output = slim.batch_norm(output, scope="bn2")
    output = slim.dropout(output, 0.5, scope="dropout2")
    output = slim.fully_connected(
        output, vocab_size, activation_fn=tf.nn.sigmoid, scope="fc3")
    return {"predictions": output}
##########################################################################################################
class MLPA(models.BaseModel): # Model3nn4096BnReluDropSkipDoubleLr1e7decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=4096, l2_penalty=1e-7, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1152, 4096], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)

    output = slim.fully_connected(
        input_projected_plus_h2, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_3')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}
    #return {"predictions": output}



class MLPB(models.BaseModel): # Model4nn2048BnReluDropSkipDoubleLr1e7decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=2048, l2_penalty=1e-7, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1152, 2048], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)


    hidden3 = tf.layers.dense(
        inputs=input_projected_plus_h2, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_3')

    bn3 = tf.contrib.layers.batch_norm(inputs=hidden3, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn3')

    relu3 = tf.nn.relu(hidden3, name=prefix+'relu3' )

    dropout3 = tf.layers.dropout(inputs=relu3, rate=0.5, name=prefix+"dropout3")


    input_projected_plus_h3 = tf.add(input_projected, dropout3)
    #input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

    output = slim.fully_connected(
        input_projected_plus_h3, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_4')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}
    #return {"predictions": output}


class MLPC(models.BaseModel): # Model4nn4096BnReluDropSkipDoubleLr1e7decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=4096, l2_penalty=1e-7, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1152, 4096], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)


    hidden3 = tf.layers.dense(
        inputs=input_projected_plus_h2, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_3')

    bn3 = tf.contrib.layers.batch_norm(inputs=hidden3, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn3')

    relu3 = tf.nn.relu(hidden3, name=prefix+'relu3' )

    dropout3 = tf.layers.dropout(inputs=relu3, rate=0.5, name=prefix+"dropout3")


    input_projected_plus_h3 = tf.add(input_projected, dropout3)
    #input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

    output = slim.fully_connected(
        input_projected_plus_h3, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_4')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}
    #return {"predictions": output}






class MLPD(models.BaseModel): # Model4nn4096BnReluDropSkipDoubleLr5e7decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=4096, l2_penalty=5e-7, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1152, 4096], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)


    hidden3 = tf.layers.dense(
        inputs=input_projected_plus_h2, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_3')

    bn3 = tf.contrib.layers.batch_norm(inputs=hidden3, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn3')

    relu3 = tf.nn.relu(hidden3, name=prefix+'relu3' )

    dropout3 = tf.layers.dropout(inputs=relu3, rate=0.5, name=prefix+"dropout3")


    input_projected_plus_h3 = tf.add(input_projected, dropout3)
    #input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

    output = slim.fully_connected(
        input_projected_plus_h3, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_4')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}
    #return {"predictions": output}


class MLPE(models.BaseModel): # Model4nn4096BnReluDropSkipDoubleLr1e6decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=4096, l2_penalty=1e-6, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1152, 4096], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)


    hidden3 = tf.layers.dense(
        inputs=input_projected_plus_h2, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_3')

    bn3 = tf.contrib.layers.batch_norm(inputs=hidden3, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn3')

    relu3 = tf.nn.relu(hidden3, name=prefix+'relu3' )

    dropout3 = tf.layers.dropout(inputs=relu3, rate=0.5, name=prefix+"dropout3")


    input_projected_plus_h3 = tf.add(input_projected, dropout3)
    #input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

    output = slim.fully_connected(
        input_projected_plus_h3, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_4')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}
    #return {"predictions": output}
##########################################################################################################################
class EnsembleModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = LogisticModel().create_model(model_input, vocab_size)
    output_b = MLPModel().create_model(model_input, vocab_size)
    output_c = MoeModel().create_model(model_input, vocab_size, num_mixtures=7)
    output_d = DeepSkip().create_model(model_input, vocab_size)

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]
    t3 = output_c["predictions"]
    t4 = output_d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])

    scalar = tf.constant(0.25)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class ReducedMoeModel(models.BaseModel):
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
    expert_mid_activations = slim.fully_connected(
        model_input,
        1024,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="expertsMid")
    expert_activations = slim.fully_connected(
        expert_mid_activations,
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

###############################################################################################################################
class MLPAverageA(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = MLPE().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = MLPE().create_model(model_input, vocab_size, prefix='u2048b/')
    output_c = MLPE().create_model(model_input, vocab_size, prefix='u2048c/')
    output_d = MLPE().create_model(model_input, vocab_size, prefix='u2048d/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]
    t3 = output_c["predictions"]
    t4 = output_d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])

    scalar = tf.constant(0.25)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class MLPAverageB(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = MLPE().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = MLPE().create_model(model_input, vocab_size, prefix='u2048b/')
    output_c = MLPE().create_model(model_input, vocab_size, prefix='u2048c/')
    output_d = MLPE().create_model(model_input, vocab_size, prefix='u2048d/')
    output_e = MLPE().create_model(model_input, vocab_size, prefix='u2048e/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]
    t3 = output_c["predictions"]
    t4 = output_d["predictions"]
    t5 = output_e["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4, t5])

    scalar = tf.constant(0.20)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class MLPAverageC(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = MLPE().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = MLPE().create_model(model_input, vocab_size, prefix='u2048b/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]

    output_sum = tf.add_n([t1, t2])

    scalar = tf.constant(0.50)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class MLPESoftmax(models.BaseModel): # Model4nn4096BnReluDropSkipDoubleLr1e6decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=4096, l2_penalty=1e-6, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1152, 4096], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)


    hidden3 = tf.layers.dense(
        inputs=input_projected_plus_h2, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_3')

    bn3 = tf.contrib.layers.batch_norm(inputs=hidden3, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn3')

    relu3 = tf.nn.relu(hidden3, name=prefix+'relu3' )

    dropout3 = tf.layers.dropout(inputs=relu3, rate=0.5, name=prefix+"dropout3")


    input_projected_plus_h3 = tf.add(input_projected, dropout3)
    #input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

    output = slim.fully_connected(
        input_projected_plus_h3, vocab_size, activation_fn=tf.nn.softmax,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_4')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}

class MLPEUse(models.BaseModel): # Model4nn4096BnReluDropSkipDoubleLr1e6decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=4096, l2_penalty=1e-6, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[4176, 4096], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)


    hidden3 = tf.layers.dense(
        inputs=input_projected_plus_h2, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_3')

    bn3 = tf.contrib.layers.batch_norm(inputs=hidden3, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn3')

    relu3 = tf.nn.relu(hidden3, name=prefix+'relu3' )

    dropout3 = tf.layers.dropout(inputs=relu3, rate=0.5, name=prefix+"dropout3")


    input_projected_plus_h3 = tf.add(input_projected, dropout3)
    #input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

    output = slim.fully_connected(
        input_projected_plus_h3, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_4')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}

class MLPESmall(models.BaseModel): # Model4nn4096BnReluDropSkipDoubleLr1e6decay9

  def create_model(self, model_input, vocab_size, num_hidden_units=3072, l2_penalty=1e-6, prefix='', **unused_params):
    """Creates a logistic model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    # Initialize weights for projection
    w_s = tf.Variable(tf.random_normal(shape=[1152, 3072], stddev=0.01))
    input_projected = tf.matmul(model_input, w_s)

    hidden1 = tf.layers.dense(
        inputs=model_input, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_1')


    bn1 = tf.contrib.layers.batch_norm(inputs=hidden1, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn1')

    relu1 = tf.nn.relu(hidden1, name=prefix+'relu1' )

    dropout1 = tf.layers.dropout(inputs=relu1, rate=0.5, name=prefix+"dropout1")


    hidden2 = tf.layers.dense(
        inputs=dropout1, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_2')

    bn2 = tf.contrib.layers.batch_norm(inputs=hidden2, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn2')


    relu2 = tf.nn.relu(hidden2, name=prefix+'relu2' )

    dropout2 = tf.layers.dropout(inputs=relu2, rate=0.5, name=prefix+"dropout2")

    input_projected_plus_h2 = tf.add(input_projected, dropout2)


    hidden3 = tf.layers.dense(
        inputs=input_projected_plus_h2, units=num_hidden_units, activation=None,
        kernel_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix+'fc_3')

    bn3 = tf.contrib.layers.batch_norm(inputs=hidden3, decay=0.9, center=True,
                                scale=True, epsilon=1e-7, activation_fn=None,
                                is_training=True, scope=prefix+'bn3')

    relu3 = tf.nn.relu(hidden3, name=prefix+'relu3' )

    dropout3 = tf.layers.dropout(inputs=relu3, rate=0.5, name=prefix+"dropout3")


    input_projected_plus_h3 = tf.add(input_projected, dropout3)
    #input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

    output = slim.fully_connected(
        input_projected_plus_h3, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix+'fc_4')


    weights_norm = tf.add_n(tf.losses.get_regularization_losses())

    return {"predictions": output, "regularization_loss": weights_norm}

class ComplexMoeAverageA(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048b/')
    output_c = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048c/')
    output_d = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048d/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]
    t3 = output_c["predictions"]
    t4 = output_d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])

    scalar = tf.constant(0.25)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class ComplexMoeAverageB(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048b/')
    output_c = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048c/')
    output_d = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048d/')
    output_e = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048e/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]
    t3 = output_c["predictions"]
    t4 = output_d["predictions"]
    t5 = output_e["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4, t5])

    scalar = tf.constant(0.20)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class ComplexMoeAverageC(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = ComplexMoeModel().create_model(model_input, vocab_size, prefix='u2048b/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]

    output_sum = tf.add_n([t1, t2])

    scalar = tf.constant(0.50)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class MoeAverageA(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = MoeModel().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = MoeModel().create_model(model_input, vocab_size, prefix='u2048b/')
    output_c = MoeModel().create_model(model_input, vocab_size, prefix='u2048c/')
    output_d = MoeModel().create_model(model_input, vocab_size, prefix='u2048d/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]
    t3 = output_c["predictions"]
    t4 = output_d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])

    scalar = tf.constant(0.25)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class MoeAverageB(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates an Average prediction of NN models.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_a = MoeModel().create_model(model_input, vocab_size, prefix='u2048a/')
    output_b = MoeModel().create_model(model_input, vocab_size, prefix='u2048b/')
    output_c = MoeModel().create_model(model_input, vocab_size, prefix='u2048c/')
    output_d = MoeModel().create_model(model_input, vocab_size, prefix='u2048d/')
    output_e = MoeModel().create_model(model_input, vocab_size, prefix='u2048e/')

    t1 = output_a["predictions"]
    t2 = output_b["predictions"]
    t3 = output_c["predictions"]
    t4 = output_d["predictions"]
    t5 = output_e["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4, t5])

    scalar = tf.constant(0.20)
    output = tf.scalar_mul(scalar, output_sum)

    return {"predictions": output}

class StackEnsembleA(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a Stack of Neural Networks Model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_2048a = LogisticModel().create_model(model_input, vocab_size)
    output_2048b = MoeModel().create_model(model_input, vocab_size)
    output_2048c = MLPE().create_model(model_input, vocab_size)
    output_2048d = BiggerNN().create_model(model_input, vocab_size)

    t1 = output_2048a["predictions"]
    t2 = output_2048b["predictions"]
    t3 = output_2048c["predictions"]
    t4 = output_2048d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])
    scalar = tf.constant(0.25)
    avg_output = tf.scalar_mul(scalar, output_sum)
    stacked_fc1 = slim.fully_connected(
      avg_output,
      4096,
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="Stack/fc1")
    stacked_fc1 = slim.dropout(stacked_fc1, 0.5, scope="Stack/dropout1")
    stacked_fc2 = slim.fully_connected(
      stacked_fc1,
      vocab_size,
      activation_fn=tf.nn.sigmoid,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="Stack/fc2")

    return {"predictions": stacked_fc2}

class StackEnsembleB(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a Stack of Neural Networks Model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_2048a = LogisticModel().create_model(model_input, vocab_size)
    output_2048b = MoeModel().create_model(model_input, vocab_size)
    output_2048c = MLPE().create_model(model_input, vocab_size)
    output_2048d = ComplexMoeModel().create_model(model_input, vocab_size)

    t1 = output_2048a["predictions"]
    t2 = output_2048b["predictions"]
    t3 = output_2048c["predictions"]
    t4 = output_2048d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])
    scalar = tf.constant(0.25)
    avg_output = tf.scalar_mul(scalar, output_sum)

    output = MLPEUse().create_model(avg_output, vocab_size)

    return {"predictions": output}

class StackEnsembleC(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a Stack of Neural Networks Model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output_2048a = LogisticModel().create_model(model_input, vocab_size)
    output_2048b = MoeModel().create_model(model_input, vocab_size)
    output_2048c = MLPE().create_model(model_input, vocab_size)
    output_2048d = BiggerNN().create_model(model_input, vocab_size)

    t1 = output_2048a["predictions"]
    t2 = output_2048b["predictions"]
    t3 = output_2048c["predictions"]
    t4 = output_2048d["predictions"]

    output_sum = tf.add_n([t1, t2, t3, t4])
    scalar = tf.constant(0.25)
    avg_output = tf.scalar_mul(scalar, output_sum)

    output = MLPEUse().create_model(avg_output, vocab_size)

    return {"predictions": output}
