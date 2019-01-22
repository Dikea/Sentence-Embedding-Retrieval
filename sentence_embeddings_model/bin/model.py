#!/bin/env python
#-*- encoding: utf-8 -*- 


import sys
sys.path.append('../')

import time
import os
import pickle as pkl
import numpy as np
import shutil
import tensorflow as tf
import config
from tools.log import g_log_inst as logger
from tools.data_helper import Helper
import tools.config


#os.environ['CUDA_VISIBLE_DEVICES'] = ''


# Model paras
class Infersent_para(object):
    """Build parameters"""
    embedding_size = tools.config.embeddings_size
    hidden_size = 256
    hidden_layers = 1
    batch_size = 64
    keep_prob_dropout = 1.0
    learning_rate = 0.1 
    bidirectional = True
    decay = 0.99
    lrshrink = 5
    eval_step = 500
    uniform_init_scale = 0.1
    clip_gradient_norm = 5.0
    save_every = 1000000
    epochs = 5 

    # Settings for debug
    debug_mode = False
    if debug_mode:
        train_steps = 10
        dev_steps = 1 
        eval_step = 10 
        epochs = 2


class Infersent_model(object):

    def __init__(self, parameters, path):
        self.para = parameters
        self.path = path
        self.learning_rate = self.para.learning_rate

        logger.get().info('Building Graph')
        self.graph = tf.get_default_graph()
        '''
        self.initializer = tf.contrib.layers.xavier_initializer(
            uniform=True, seed=None, dtype=tf.float32)
        self.uniform_initializer = tf.random_uniform_initializer(
            minval = -self.para.uniform_init_scale, 
            maxval = self.para.uniform_init_scale)
        '''

        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)
        self.eta = tf.placeholder(tf.float32, 
            [],
            name = "eta")
        # Bilstm input data, [batch_size, lstm_num_steps, embedding_size]
        self.s1_embedded = tf.placeholder(tf.float32, 
            [None, None, self.para.embedding_size], 
            "s1_embedded")
        self.s1_lengths = tf.placeholder(tf.int32, 
            [None], 
            "s1_embedded")
        self.s2_embedded = tf.placeholder(tf.float32, 
            [None, None, self.para.embedding_size], 
            "s2_embedded")
        self.s2_lengths = tf.placeholder(tf.int32, 
            [None], 
            "s2_embedded")
        self.labels = tf.placeholder(tf.int32, 
            [None], 
            "labels")

        def get_lstm_cell(lstm_size, keep_prob_dropout):
            lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob =
                                                      keep_prob_dropout)
            return lstm_cell

        with tf.variable_scope("encoder") as varscope:

            cells_fw = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(
                                                    self.para.hidden_size, 
                                                    self.para.keep_prob_dropout)
                                                    for _ in range(self.para.hidden_layers)])
            cells_bw = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(
                                                    self.para.hidden_size, 
                                                    self.para.keep_prob_dropout)
                                                    for _ in range(self.para.hidden_layers)])
            
            s1_sentences_states, s1_last_state = tf.nn.bidirectional_dynamic_rnn(
                cells_fw, cells_bw, 
                inputs = self.s1_embedded, 
                sequence_length = self.s1_lengths, 
                dtype = tf.float32, 
                scope = varscope)

            s1_states_fw, s1_states_bw = s1_sentences_states
            self.s1_states_h = tf.concat([s1_states_fw, s1_states_bw], axis = 2)
            self.s1_states_h = tf.reduce_max(self.s1_states_h, axis = 1)

            varscope.reuse_variables()

            s2_sentences_states, s2_last_state = tf.nn.bidirectional_dynamic_rnn(
                cells_fw, cells_bw, 
                inputs = self.s2_embedded, 
                sequence_length = self.s2_lengths, 
                dtype = tf.float32, 
                scope = varscope)

            s2_states_fw, s2_states_bw = s2_sentences_states
            # Concatenate forward state and backward state
            self.s2_states_h = tf.concat([s2_states_fw, s2_states_bw], axis = 2)
            # Max pooling
            self.s2_states_h = tf.reduce_max(self.s2_states_h, axis = 1)

        with tf.variable_scope("classification_layer") as varscope:
            self.features = tf.concat(
                [self.s1_states_h, 
                self.s2_states_h, 
                tf.abs(self.s1_states_h - self.s2_states_h), 
                self.s1_states_h * self.s2_states_h],
                axis = 1)
            hidden = tf.contrib.layers.fully_connected(
                inputs = self.features,
                num_outputs = 512)
            logits = tf.contrib.layers.fully_connected(
                inputs = hidden,
                activation_fn = None,
                num_outputs = 2)

        with tf.variable_scope("loss") as varscope:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.labels,
                logits = logits))
            self.opt_op = tf.contrib.layers.optimize_loss(
                loss = self.loss, 
                global_step = self.global_step, 
                learning_rate = self.eta, 
                optimizer = 'SGD', 
                clip_gradients=self.para.clip_gradient_norm, 
                learning_rate_decay_fn=None,
                summaries=None)

            self.loss_sum = tf.summary.scalar('loss', self.loss)
            self.lr_sum = tf.summary.scalar('learning_rate', self.eta)

        with tf.name_scope('accuracy'):
            pred = tf.argmax(logits, 1)
            correct_prediction = tf.equal(self.labels, tf.cast(pred, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.acc_sum = tf.summary.scalar('accuracy', self.accuracy)


    def initialize(self):
        self.train_loss_writer = tf.summary.FileWriter(
            self.path +'tensorboard/train_loss', self.sess.graph)
        self.dev_loss_writer = tf.summary.FileWriter(
            self.path +'tensorboard/dev_loss', self.sess.graph)
        self.dev_accuracy_writer = tf.summary.FileWriter(
            self.path +'tensorboard/dev_accuracy', self.sess.graph)
        self.train_summary = tf.summary.merge([self.loss_sum, self.lr_sum])
        self.dev_loss_summary = tf.summary.merge([self.loss_sum])
        self.saver = tf.train.Saver()
        self.start_time = time.time()
        self.dev_accuracies = []
        self.dev_loss = []


    def train(self, train_data, dev_data):
        try:
            logger.get().info('Starting training')
            self.dev_accuracy = -np.inf
            for epoch in range(self.para.epochs):
                self._run_epoch(epoch, train_data, dev_data)
                logger.get().info('End of epoch %d, Time elapsed: %.4f hours', 
                    epoch, (time.time() - self.start_time) / 3600)
        except KeyboardInterrupt:
            self.save_model(self.path + '/saved_models/tmp/', 
                self.global_step.eval(session = self.sess))

    
    def _run_epoch(self, epoch, train_data, dev_data):
        logger.get().info('Epoch %d, shuffling data...', epoch)
        '''
        self.learning_rate = (self.learning_rate * self.para.decay 
                              if epoch > 0 else self.learning_rate) 
        logger.get().debug('Learning rate in epoch %d is: %.6f',
                           epoch, self.learning_rate)
        '''
        batch_time = time.time()
        train_loss = 0
        train_length = len(train_data['s1'])  
        perm = np.random.permutation(train_length)
        train_s1 = np.array(train_data['s1'])[perm]
        train_s2 = np.array(train_data['s2'])[perm]
        train_targets = np.array(train_data['label'])[perm]
        train_steps = self.para.train_steps if self.para.debug_mode else \
            (train_length // self.para.batch_size)
        logger.get().debug('train_steps: %d', train_steps)

        dev_steps = self.para.dev_steps if self.para.debug_mode else 40 
        for train_step in range(train_steps):
            batch_loss, current_step = self._batch_train(
                train_step, train_s1, train_s2, train_targets)
            train_loss += batch_loss/self.para.eval_step
            if current_step % self.para.eval_step == 0:
                logger.get().debug('Average training loss at epoch %d step %d: %.6f',
                                   epoch, current_step, train_loss)
                dev_accuracy, dev_loss, dev_loss_summary = ( 
                    self._evaluate(dev_data, dev_steps))
                self.dev_loss_writer.add_summary(dev_loss_summary, current_step)
                logger.get().info(
                    'Average (across %d data points) dev loss at epoch %d step %d: %.6f',
                    dev_steps * self.para.batch_size, epoch, current_step, dev_loss)
                logger.get().info('Accuracy: %0.2f', dev_accuracy)
                end_time = time.time()
                logger.get().info('Time for %d steps: %0.2f seconds',
                                  self.para.eval_step, end_time - batch_time)
                batch_time = time.time()
            if current_step % self.para.save_every == 0:
                self.save_model(self.path + '/saved_models/', current_step)

        # Evaluate accuracy on development data
        dev_accuracy, dev_loss, dev_loss_summary = (
            self._evaluate(dev_data))
        self.dev_accuracies.append(dev_accuracy)
        self.dev_loss.append(dev_loss)
        np.save(self.path + 'dev_accuracies.npy', np.array(self.dev_accuracies))
        np.save(self.path + 'dev_loss.npy', np.array(self.dev_loss))
        logger.get().info('Current dev accuracy: %0.3f, Previous best dev accuracy: %0.3f',
            dev_accuracy, self.dev_accuracy)
        '''
        if dev_accuracy > self.dev_accuracy:
            # Save best model
            self.export_serving_model()
            self.dev_accuracy = dev_accuracy
            logger.get().info('Dev accuracy improved')
        else:
            self.learning_rate = self.learning_rate/self.para.lrshrink
            logger.get().info('Dev accuracy didnt improve, new learning rate: %0.6f', 
                              self.learning_rate)
        '''
        if epoch == 0:
            self.dev_accuracy = dev_accuracy
            self.export_serving_model()
        if (dev_accuracy > self.dev_accuracy) and (epoch > 0):
            self.export_serving_model()
            self.learning_rate = self.learning_rate/self.para.lrshrink
            self.dev_accuracy = dev_accuracy
            logger.get().info('Dev accuracy improved, new learning rate: %0.6f',
                self.learning_rate)
        else:
            self.learning_rate = self.learning_rate * self.para.decay
            logger.get().info('Dev accuracy didnt improve, new learning rate: %0.6f', 
                self.learning_rate)

        # Save trained model after every finished   
        self.save_model(self.path + '/saved_models/', 
                        self.global_step.eval(session = self.sess))

    
    def _batch_train(self, train_step, train_s1, train_s2, train_targets):
        begin = train_step * self.para.batch_size
        end = (train_step + 1) * self.para.batch_size
        batch_s1, batch_s1_len = Helper.get_batch(train_s1[begin: end])
        batch_s2, batch_s2_len = Helper.get_batch(train_s2[begin: end])
        batch_labels = train_targets[begin : end]

        train_dict = {
            self.s1_embedded: batch_s1,
            self.s1_lengths: batch_s1_len, 
            self.s2_embedded: batch_s2,
            self.s2_lengths: batch_s2_len, 
            self.labels: batch_labels.T,
            self.eta: self.learning_rate}

        _, batch_loss, current_step, batch_summary = self.sess.run(
            [self.opt_op, self.loss, self.global_step, self.train_summary], 
            feed_dict=train_dict)

        logger.get().debug('Step %d loss: %0.5f', current_step, batch_loss)
        self.train_loss_writer.add_summary(batch_summary, current_step)
        return batch_loss, current_step


    def _evaluate(self, dev_data, dev_steps = None):
        dev_accuracy = 0
        dev_loss = 0
        dev_length = len(dev_data['s1'])
        perm = np.random.permutation(dev_length)
        dev_s1 = np.array(dev_data['s1'])[perm]
        dev_s2 = np.array(dev_data['s2'])[perm]
        dev_targets = np.array(dev_data['label'])[perm]
        dev_steps = (dev_length // self.para.batch_size if dev_steps is None
                     else dev_steps)
        logger.get().debug('evaluate dev_steps: %d', dev_steps)
        for dev_step in range(dev_steps):
            begin = dev_step * self.para.batch_size
            end = (dev_step + 1) * self.para.batch_size
            batch_s1, batch_s1_len = Helper.get_batch(dev_s1[begin: end])
            batch_s2, batch_s2_len = Helper.get_batch(dev_s2[begin: end])
            batch_labels = dev_targets[begin : end]

            dev_dict = {
                self.s1_embedded: batch_s1,
                self.s1_lengths: batch_s1_len, 
                self.s2_embedded: batch_s2,
                self.s2_lengths: batch_s2_len, 
                self.labels: batch_labels.T}

            batch_accuracy, batch_loss, dev_loss_summary = self.sess.run(
                [self.accuracy, self.loss, self.dev_loss_summary], 
                feed_dict=dev_dict)
            dev_accuracy += batch_accuracy / dev_steps
            dev_loss += batch_loss / dev_steps
        return dev_accuracy, dev_loss, dev_loss_summary


    def save_model(self, path, step):
        """Save trained model's variables"""
        if not os.path.exists(path):
            os.mkdir(path)
        self.saver.save(sess = self.sess, save_path = path + 
                        '/step_%d' % step, write_state = False)
        logger.get().info('Model saved')


    def load_model(self, path, step):
        """Load trained model's variables"""
        self.sess = tf.Session(graph = self.graph)
        saver = tf.train.Saver()
        saver.restore(self.sess, path + '/saved_models/step_%d' % step)
        logger.get().info('Model restored')


    def export_serving_model(self):
        """Export model for tensor-serving"""
        export_path_base = config.serving_model_path
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(config.serving_model_version)))
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        logger.get().info('Export trained model to %s', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        inputs = {
            's_embedded': tf.saved_model.utils.build_tensor_info(self.s1_embedded),
            's_lengths': tf.saved_model.utils.build_tensor_info(self.s1_lengths)}
        outputs = {
            's_embeddings': tf.saved_model.utils.build_tensor_info(self.s1_states_h)}
        encoder_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs,
                outputs,
                method_name = 'encoder_signature'))
        builder.add_meta_graph_and_variables(
            self.sess, ['infersent_model'],
            signature_def_map = {
                'encoder': encoder_signature
            })
        builder.save()
        logger.get().info('Export model done')


def main(_):
    # Remove old log file, start log serving
    log_path = './log/model.log' 
    if os.path.exists(log_path):
        print('Remove existing log folder')
        os.remove(log_path)
    logger.start(log_path, name = __name__, level = 'DEBUG')

    # Get required path
    path = config.snli_path 
    model_path = config.model_path 
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(model_path +'tensorboard'):
        print('Remove existing tensorboard folder')
        shutil.rmtree(model_path +'tensorboard')

    # Data prepare 
    Helper.init()

    # Prepare data for train, develop and test
    train, dev, test = Helper.get_nli(path)

    # Build model and train
    paras = Infersent_para()
    tf.reset_default_graph()
    model = Infersent_model(parameters = paras, path = model_path)
    model.sess = tf.Session()
    tf.global_variables_initializer().run(session = model.sess)
    model.initialize()
    model.train(train, dev)


if __name__ == '__main__':
    tf.app.run()
