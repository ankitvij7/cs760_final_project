# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in training mode.

This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.

For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.

Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100

  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \
                                --train_vggish=False \
                                --checkpoint /path/to/model/checkpoint
"""

from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow as tf
import csv
import vggish_input
import vggish_params
import vggish_slim

slim = tf.contrib.slim

_NUM_CLASSES = 3


def _get_examples_batch():
    """Returns a shuffled batch of examples of all audio classes.

    Note that this is just a toy function because this is a simple demo intended
    to illustrate how the training code might work.

    Returns:
      a tuple (features, labels) where features is a NumPy array of shape
      [batch_size, num_frames, num_bands] where the batch_size is variable and
      each row is a log mel spectrogram patch of shape [num_frames, num_bands]
      suitable for feeding VGGish, while labels is a NumPy array of shape
      [batch_size, num_classes] where each row is a multi-hot label vector that
      provides the labels for corresponding rows in features.
    """
    # Make a waveform for each class.
    num_seconds = 5
    sr = 44100  # Sampling rate.
    t = np.linspace(0, num_seconds, int(num_seconds * sr))  # Time axis.
    # Random sine wave.
    freq = np.random.uniform(100, 1000)
    sine = np.sin(2 * np.pi * freq * t)
    # Random constant signal.
    magnitude = np.random.uniform(-1, 1)
    const = magnitude * t
    # White noise.
    noise = np.random.normal(-1, 1, size=t.shape)

    # Make examples of each signal and corresponding labels.
    # Sine is class index 0, Const class index 1, Noise class index 2.
    sine_examples = vggish_input.waveform_to_examples(sine, sr)
    sine_labels = np.array([[1, 0, 0]] * sine_examples.shape[0])
    const_examples = vggish_input.waveform_to_examples(const, sr)
    const_labels = np.array([[0, 1, 0]] * const_examples.shape[0])
    noise_examples = vggish_input.waveform_to_examples(noise, sr)
    noise_labels = np.array([[0, 0, 1]] * noise_examples.shape[0])

    print(sine_examples.shape)
    print(sine_labels.shape)

    # Shuffle (example, label) pairs across all classes.
    all_examples = np.concatenate((sine_examples, const_examples, noise_examples))
    print(all_examples.shape)
    all_labels = np.concatenate((sine_labels, const_labels, noise_labels))
    print(all_labels.shape)
    labeled_examples = list(zip(all_examples, all_labels))
    print(len(labeled_examples))
    shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    print(features[0].shape)
    print(labels[0].shape)
    return (features, labels)


NUM_CLASSES = 4
mood_labels = {
    "Happy music": [[1, 0, 0, 0]],
    "Sad music": [[0, 1, 0, 0]],
    "Angry music": [[0, 0, 1, 0]],
    "Scary music": [[0, 0, 0, 1]]
}
input_file = '../dataset/moods_unbalanced_100each.csv'
input_wav_suffix = '.wav'
input_wav_prefix = 'wav_outputs_unbalanced/'


def get_examples_batch(batch_num, num_batches):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile, quotechar='"', delimiter=' ', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        # get record count
        num_rows = len(list(reader))
        row_cnt = 0;
        batch_size = int(round(num_rows / num_batches))
        start_row = batch_num * batch_size
        end_row = start_row + batch_size
        # reset the reader.
        csvfile.seek(0)

        labeled_examples = []
        for row in reader:
            # print('row: %d, s: %d,  e:%d, '% (row_cnt, start_row, end_row))
            if row_cnt >= start_row and row_cnt < end_row:
                # process this row.
                input_wav = input_wav_prefix + row[3] + input_wav_suffix
                print("Row %d, Load: %s" % (row_cnt, input_wav))
                # read in this file, and convert to spectrograph, and build up our features.
                example = vggish_input.wavfile_to_examples(input_wav)
                # fetch our label, and build up our labels.
                label = np.array(mood_labels[row[4]] * example.shape[0])
                labeled_examples = labeled_examples + list(zip(example, label))
                # print(len(labeled_examples))
                # print(labeled_examples[0][0].shape)
                # print(labeled_examples[0][1].shape)

            # next row...
            row_cnt = row_cnt + 1

        shuffle(labeled_examples)

        # Separate and return the features and labels.
        features = [example for (example, _) in labeled_examples]
        labels = [label for (_, label) in labeled_examples]
        # print(len(labeled_examples))
        # print(features[0].shape)
        # print(labels[0].shape)
        return (features, labels)


def vggish_train(checkpoint, num_batches):
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(True)  # Train the model.

        # Define a shallow classification model and associated training ops on top of VGGish.
        with tf.variable_scope('mymodel'):
            # Add a fully connected layer with 100 units.
            num_units = 100
            fc = slim.fully_connected(embeddings, num_units)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(fc, NUM_CLASSES, activation_fn=None, scope='logits')
            tf.sigmoid(logits, name='prediction')

            # Add training ops.
            with tf.variable_scope('train'):
                global_step = tf.Variable(0, name='global_step', trainable=False,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='labels')

                # Cross-entropy label loss.
                xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)

                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(learning_rate=vggish_params.LEARNING_RATE, epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')

        # Initialize all variables in the model, and then create a saver.
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        # The training loop.
        batch_num = 0
        for _ in range(num_batches):
            (features, labels) = get_examples_batch(batch_num, num_batches)
            [num_steps, loss, _] = sess.run([global_step_tensor, loss_tensor, train_op], feed_dict={features_tensor: features, labels_tensor: labels})
            print('Step %d: loss %g' % (num_steps, loss))
            batch_num = batch_num + 1

        # Save off our model before leaving.
        save_path = saver.save(sess, checkpoint)
        print('Model saved in path: %s' % save_path)


# checkpoint = "vgg_model_gen.ckpt"
# num_batches = 30
vggish_train('./gen/vgg_model_gen.ckpt', 20)
# vggish_train('./vgg_model_gen.ckpt', 1)
# get_examples_batch(1, 20)
# _get_examples_batch()
