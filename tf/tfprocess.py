#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import keras_cv
import random
import tensorflow as tf
import time
import bisect
import lc0_az_policy_map
import attention_policy_map as apm
import proto.net_pb2 as pb
from functools import reduce
import operator

from net import Net


def square_relu(x):
    return tf.nn.relu(x)**2


class Gating(tf.keras.layers.Layer):

    def __init__(self, name=None, additive=True, init_value=None, **kwargs):
        self.additive = additive
        if init_value is None:
            init_value = 0 if self.additive else 1
        self.init_value = init_value
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.gate = self.add_weight(name='gate',
                                    shape=input_shape[1:],
                                    constraint=tf.keras.constraints.NonNeg()
                                    if not self.additive else None,
                                    initializer=tf.constant_initializer(
                                        self.init_value),
                                    trainable=True)

    def call(self, inputs):
        return tf.add(inputs, self.gate) if self.additive else tf.multiply(
            inputs, self.gate)


def ma_gating(inputs, name):
    out = Gating(name=name + '/mult_gate', additive=False)(inputs)
    out = Gating(name=name + '/add_gate', additive=True)(out)
    return out


class ApplySqueezeExcitation(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
                                            [-1, self.reshape_size, 1, 1]),
                                 2,
                                 axis=1)
        return tf.nn.sigmoid(gammas) * x + betas


class ApplyPolicyMap(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplyPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs):
        h_conv_pol_flat = tf.reshape(inputs, [-1, 64 * 8 * 8])
        return tf.matmul(h_conv_pol_flat,
                         tf.cast(self.fc1, h_conv_pol_flat.dtype))


class Metric:

    def __init__(self, short_name, long_name, suffix='', **kwargs):
        self.short_name = short_name
        self.long_name = long_name
        self.suffix = suffix
        self.value = 0.0
        self.count = 0

    def assign(self, value):
        self.value = value
        self.count = 1

    def accumulate(self, value):
        if self.count > 0:
            self.value = self.value + value
            self.count = self.count + 1
        else:
            self.assign(value)

    def merge(self, other):
        assert self.short_name == other.short_name
        self.value = self.value + other.value
        self.count = self.count + other.count

    def get(self):
        if self.count == 0:
            return self.value
        return self.value / self.count

    def reset(self):
        self.value = 0.0
        self.count = 0


class TFProcess:

    def __init__(self, cfg):
        self.cfg = cfg
        self.net = Net()
        self.root_dir = os.path.join(self.cfg['training']['path'], self.cfg['name'])

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model'].get('filters', 0)
        self.RESIDUAL_BLOCKS = self.cfg['model'].get('residual_blocks', 0)
        self.SE_ratio = self.cfg['model'].get('se_ratio', 0)
        self.encoder_heads = self.cfg['model'].get('encoder_heads', 2)

        self.policy_channels = self.cfg['model'].get('policy_channels', 32)

        precision = self.cfg['training'].get('precision', 'single')
        loss_scale = self.cfg['training'].get('loss_scale', 128)
        self.virtual_batch_size = self.cfg['model'].get(
            'virtual_batch_size', None)

        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else loss_scale

        policy_head = self.cfg['model'].get('policy', 'convolution')
        moves_left_head = self.cfg['model'].get('moves_left', 'none')
        input_mode = self.cfg['model'].get('input_type', 'classic')
        default_activation = self.cfg['model'].get('default_activation', 'relu')

        self.POLICY_HEAD = None
        self.MOVES_LEFT_HEAD = None
        self.INPUT_MODE = None
        self.DEFAULT_ACTIVATION = None

        if policy_head == "classical":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CONVOLUTION
        else:
            raise ValueError(
                "Unknown policy head format: {}".format(policy_head))

        self.net.set_policyformat(self.POLICY_HEAD)

        if moves_left_head == "none":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_NONE
            self.moves_left = False
        elif moves_left_head == "v1":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_V1
            self.moves_left = True
        else:
            raise ValueError(
                "Unknown moves left head format: {}".format(moves_left_head))

        self.net.set_movesleftformat(self.MOVES_LEFT_HEAD)

        if input_mode == "classic":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
        elif input_mode == "frc_castling":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
        elif input_mode == "canonical":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
        elif input_mode == "canonical_100":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
        elif input_mode == "canonical_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
        elif input_mode == "canonical_v2":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
        elif input_mode == "canonical_v2_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
        else:
            raise ValueError(
                "Unknown input mode format: {}".format(input_mode))

        self.net.set_input(self.INPUT_MODE)

        if default_activation == "relu":
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_RELU)
            self.DEFAULT_ACTIVATION = 'relu'
        elif default_activation == "mish":
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_MISH)
            try:
                self.DEFAULT_ACTIVATION = tf.keras.activations.mish
            except AttributeError:
                import tensorflow_addons as tfa
                self.DEFAULT_ACTIVATION = tfa.activations.mish
        else:
            raise ValueError("Unknown default activation type: {}".format(
                default_activation))

        self.swa_enabled = self.cfg['training'].get('swa', False)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg['training'].get('swa_max_n', 0)

        self.renorm_enabled = self.cfg['training'].get('renorm', False)
        self.renorm_max_r = self.cfg['training'].get('renorm_max_r', 1)
        self.renorm_max_d = self.cfg['training'].get('renorm_max_d', 0)
        self.renorm_momentum = self.cfg['training'].get(
            'renorm_momentum', 0.99)

        if self.cfg['gpu'] == 'all':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.strategy = tf.distribute.MirroredStrategy()
            tf.distribute.experimental_set_strategy(self.strategy)
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                       'GPU')
            tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']],
                                                     True)
            self.strategy = None
        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False,
                                       dtype=tf.int64)

    def init(self, train_dataset, test_dataset, validation_dataset=None):
        if self.strategy is not None:
            self.train_dataset = self.strategy.experimental_distribute_dataset(
                train_dataset)
        else:
            self.train_dataset = train_dataset
        self.train_iter = iter(self.train_dataset)
        if self.strategy is not None:
            self.test_dataset = self.strategy.experimental_distribute_dataset(
                test_dataset)
        else:
            self.test_dataset = test_dataset
        self.test_iter = iter(self.test_dataset)
        if self.strategy is not None and validation_dataset is not None:
            self.validation_dataset = self.strategy.experimental_distribute_dataset(
                validation_dataset)
        else:
            self.validation_dataset = validation_dataset
        if self.strategy is not None:
            this = self
            with self.strategy.scope():
                this.init_net()
        else:
            self.init_net()

    def init_net(self):
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        HISTORY = 2
        input_var = tf.keras.Input(shape=(18*HISTORY, 8, 8))
        outputs = self.construct_net(input_var)
        self.model = tf.keras.Model(inputs=input_var, outputs=outputs)
        self.model.summary()

        # swa_count initialized regardless to make checkpoint code simpler.
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
        self.swa_weights = None
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_weights = [
                tf.Variable(w, trainable=False) for w in self.model.weights
            ]

        self.active_lr = tf.Variable(0.01, trainable=False)
        # All 'new' (TF 2.10 or newer non-legacy) optimizers must have learning_rate updated manually.
        self.update_lr_manually = False
        # Be sure not to set new_optimizer before TF 2.11, or unless you edit the code to specify a new optimizer explicitly.
        if self.cfg['training'].get('new_optimizer'):
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.active_lr, momentum=0.9, nesterov=True)
            self.update_lr_manually = True
        else:
            try:
                self.optimizer = tf.keras.optimizers.legacy.SGD(
                    learning_rate=lambda: self.active_lr,
                    momentum=0.9,
                    nesterov=True)
            except AttributeError:
                self.optimizer = tf.keras.optimizers.SGD(
                    learning_rate=lambda: self.active_lr,
                    momentum=0.9,
                    nesterov=True)
        self.orig_optimizer = self.optimizer
        try:
            self.aggregator = self.orig_optimizer.aggregate_gradients
        except AttributeError:
            self.aggregator = self.orig_optimizer.gradient_aggregator
        if self.loss_scale != 1:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer, False, self.loss_scale) #TODO: revisit and probably let dynamic=True
        if self.cfg['training'].get('lookahead_optimizer'):
            import tensorflow_addons as tfa
            self.optimizer = tfa.optimizers.Lookahead(self.optimizer)

        def correct_policy(target, output):
            output = tf.cast(output, tf.float32)
            # Calculate loss on policy head
            if self.cfg['training'].get('mask_legal_moves'):
                # extract mask for legal moves from target policy
                move_is_legal = tf.greater_equal(target, 0)
                # replace logits of illegal moves with large negative value (so that it doesn't affect policy of legal moves) without gradient
                illegal_filler = tf.zeros_like(output) - 1.0e10
                output = tf.where(move_is_legal, output, illegal_filler)
            # y_ still has -1 on illegal moves, flush them to 0
            target = tf.nn.relu(target)
            return target, output

        def policy_loss(target, output):
            target, output = correct_policy(target, output)
            policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_loss_fn = policy_loss

        def policy_accuracy(target, output):
            target, output = correct_policy(target, output)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.policy_accuracy_fn = policy_accuracy

        def policy_entropy(target, output):
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            return tf.math.negative(
                tf.reduce_mean(
                    tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                                  axis=1)))

        self.policy_entropy_fn = policy_entropy

        def policy_uniform_loss(target, output):
            uniform = tf.where(tf.greater_equal(target, 0),
                               tf.ones_like(target), tf.zeros_like(target))
            balanced_uniform = uniform / tf.reduce_sum(
                uniform, axis=1, keepdims=True)
            target, output = correct_policy(target, output)
            policy_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(balanced_uniform),
                                                        logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_uniform_loss_fn = policy_uniform_loss

        pol_loss_w = self.cfg['training']['policy_loss_weight']

        reg_term_w = self.cfg['training'].get('reg_term_weight', 1.0)

        def _lossMix(policy, reg_term):
            return pol_loss_w * policy + reg_term_w * reg_term

        self.lossMix = _lossMix

        def accuracy(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.accuracy_fn = accuracy

        # Order must match the order in process_inner_loop
        self.train_metrics = [
            Metric('Policy loss', 'Policy Loss'),
            Metric('Reg', 'Reg term'),
            Metric('Total', 'Total Loss'),
        ]
        self.time_start = None
        self.last_steps = None

        # Order must match the order in calculate_test_summaries_inner_loop
        self.test_metrics = [
            Metric('Policy loss', 'Policy Loss'),
            Metric('Policy accuracy', 'Policy Accuracy', suffix='%'),
            Metric('Policy entropy', 'Policy Entropy'),
            Metric('Policy u-loss', 'Policy UL'),
        ]

        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
        self.lr = self.cfg['training']['lr_values'][0]
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-test".format(self.cfg['name'])))
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-train".format(self.cfg['name'])))
        if vars(self).get('validation_dataset', None) is not None:
            self.validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-validation".format(self.cfg['name'])))
        if self.swa_enabled:
            self.swa_writer = tf.summary.create_file_writer(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-swa-test".format(self.cfg['name'])))
            self.swa_validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-swa-validation".format(self.cfg['name'])))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer,
                                              model=self.model,
                                              global_step=self.global_step,
                                              swa_count=self.swa_count)
        self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.root_dir,
            max_to_keep=50,
            keep_checkpoint_every_n_hours=24,
            checkpoint_name=self.cfg['name'])

    def replace_weights(self, proto_filename, ignore_errors=False):
        self.net.parse_proto(proto_filename)

        filters, blocks = self.net.filters(), self.net.blocks()
        if not ignore_errors:
            if self.RESIDUAL_FILTERS != filters:
                raise ValueError("Number of filters doesn't match the network")
            if self.RESIDUAL_BLOCKS != blocks:
                raise ValueError("Number of blocks doesn't match the network")
            if self.POLICY_HEAD != self.net.pb.format.network_format.policy:
                raise ValueError("Policy head type doesn't match the network")

        # List all tensor names we need weights for.
        names = []
        for weight in self.model.weights:
            names.append(weight.name)

        new_weights = self.net.get_weights_v2(names)
        for weight in self.model.weights:
            if 'renorm' in weight.name:
                # Renorm variables are not populated.
                continue

            try:
                new_weight = new_weights[weight.name]
            except KeyError:
                error_string = 'No values for tensor {} in protobuf'.format(
                    weight.name)
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if reduce(operator.mul, weight.shape.as_list(),
                      1) != len(new_weight):
                error_string = 'Tensor {} has wrong length. Tensorflow shape {}, size in protobuf {}'.format(
                    weight.name, weight.shape.as_list(), len(new_weight))
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if weight.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if weight.name == 'input/conv2d/kernel:0' and self.net.pb.format.network_format.input < pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weight)):
                        if (i % (num_inputs * 9)) // 9 == rule50_input:
                            new_weight[i] = new_weight[i] * 99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weight.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weight.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weight.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weight, shape=weight.shape)
                weight.assign(new_weight)
        # Replace the SWA weights as well, ensuring swa accumulation is reset.
        if self.swa_enabled:
            self.swa_count.assign(tf.constant(0.))
            self.update_swa()
        # This should result in identical file to the starting one
        # self.save_leelaz_weights('restored.pb.gz')

    def restore(self):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            self.checkpoint.restore(self.manager.latest_checkpoint)

    def process_loop(self, batch_size, test_batches, batch_splits=1):
        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2
        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_size * self.cfg[
                'training'].get('num_batch_splits', 1)
            raise ValueError(
                'batch_size must be a multiple of {}'.format(required_factor))

        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg['training']['total_steps']
        for _ in range(steps % total_steps, total_steps):
            self.process(batch_size, test_batches, batch_splits=batch_splits)

    @tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    @tf.function()
    def process_inner_loop(self, x, y):
        with tf.GradientTape() as tape:
            outputs = self.model(x, training=True)
            policy = outputs #outputs[0]
            policy_loss = self.policy_loss_fn(y, policy)
            reg_term = sum(self.model.losses)
            total_loss = self.lossMix(policy_loss, reg_term)
            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)
        metrics = [
            policy_loss,
            reg_term,
            total_loss,
        ]
        return metrics, tape.gradient(total_loss, self.model.trainable_weights)

    @tf.function()
    def strategy_process_inner_loop(self, x, y):
        metrics, new_grads = self.strategy.run(self.process_inner_loop, args=(x, y))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics, new_grads

    def apply_grads(self, grads, effective_batch_splits):
        grads = [
            g[0]
            for g in self.aggregator(zip(grads, self.model.trainable_weights))
        ]
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['training'].get(
            'max_grad_norm', 10000.0) * effective_batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights),
                                       experimental_aggregate_gradients=False)
        return grad_norm

    @tf.function()
    def strategy_apply_grads(self, grads, effective_batch_splits):
        grad_norm = self.strategy.run(self.apply_grads,
                                      args=(grads, effective_batch_splits))
        grad_norm = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         grad_norm,
                                         axis=None)
        return grad_norm

    @tf.function()
    def merge_grads(self, grads, new_grads):
        return [tf.math.add(a, b) for (a, b) in zip(grads, new_grads)]

    @tf.function()
    def strategy_merge_grads(self, grads, new_grads):
        return self.strategy.run(self.merge_grads, args=(grads, new_grads))

    def train_step(self, steps, batch_size, batch_splits):
        # need to add 1 to steps because steps will be incremented after gradient update
        if (steps +
                1) % self.cfg['training']['train_avg_report_steps'] == 0 or (
                    steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.read_weights()

        # Run training for this batch
        grads = None
        for _ in range(batch_splits):
            x, y = next(self.train_iter)
            if self.strategy is not None:
                metrics, new_grads = self.strategy_process_inner_loop(x, y)
            else:
                metrics, new_grads = self.process_inner_loop(x, y)
            if not grads:
                grads = new_grads
            else:
                if self.strategy is not None:
                    grads = self.strategy_merge_grads(grads, new_grads)
                else:
                    grads = self.merge_grads(grads, new_grads)
            # Keep running averages
            for acc, val in zip(self.train_metrics, metrics):
                acc.accumulate(val)
        # Gradients of batch splits are summed, not averaged like usual, so need to scale lr accordingly to correct for this.
        effective_batch_splits = batch_splits
        if self.strategy is not None:
            effective_batch_splits = batch_splits * self.strategy.num_replicas_in_sync
        self.active_lr.assign(self.lr / effective_batch_splits)
        if self.update_lr_manually:
            self.orig_optimizer.learning_rate = self.active_lr
        if self.strategy is not None:
            grad_norm = self.strategy_apply_grads(grads,
                                                  effective_batch_splits)
        else:
            grad_norm = self.apply_grads(grads, effective_batch_splits)

        # Note: grads variable at this point has not been unscaled or
        # had clipping applied. Since no code after this point depends
        # upon that it seems fine for now.

        # Update steps.
        self.global_step.assign_add(1)
        steps = self.global_step.read_value()

        if steps % self.cfg['training'][
                'train_avg_report_steps'] == 0 or steps % self.cfg['training'][
                    'total_steps'] == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (tf.cast(steps_elapsed, tf.float32) /
                                      elapsed)
            print("step {}, lr={:g}".format(steps, self.lr), end='')
            for metric in self.train_metrics:
                print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                          metric.suffix),
                      end='')
            print(" ({:g} pos/s)".format(speed))

            after_weights = self.read_weights()
            with self.train_writer.as_default():
                for metric in self.train_metrics:
                    tf.summary.scalar(metric.long_name,
                                      metric.get(),
                                      step=steps)
                tf.summary.scalar("LR", self.lr, step=steps)
                tf.summary.scalar("Gradient norm",
                                  grad_norm / effective_batch_splits,
                                  step=steps)
                self.compute_update_ratio(before_weights, after_weights, steps)
            self.train_writer.flush()

            self.time_start = time_end
            self.last_steps = steps
            for metric in self.train_metrics:
                metric.reset()
        return steps

    def process(self, batch_size, test_batches, batch_splits):
        # Get the initial steps value before we do a training step.
        steps = self.global_step.read_value()

        # By default disabled since 0 != 10.
        if steps % self.cfg['training'].get('profile_step_freq',
                                            1) == self.cfg['training'].get(
                                                'profile_step_offset', 10):
            self.profiling_start_step = steps
            tf.profiler.experimental.start(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-profile".format(self.cfg['name'])))

        # Run test before first step to see delta since end of last run.
        if steps % self.cfg['training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps + 1):
                # Steps is given as one higher than current in order to avoid it
                # being equal to the value the end of a run is stored against.
                self.calculate_test_summaries(test_batches, steps + 1)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps + 1)

        # Determine learning rate
        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = steps % self.cfg['training']['total_steps']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1,
                                        tf.float32) / self.warmup_steps

        with tf.profiler.experimental.Trace("Train", step_num=steps):
            steps = self.train_step(steps, batch_size, batch_splits)

        if self.swa_enabled and steps % self.cfg['training']['swa_steps'] == 0:
            self.update_swa()

        # Calculate test values every 'test_steps', but also ensure there is
        # one at the final step so the delta to the first step can be calculated.
        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg[
                'training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps):
                self.calculate_test_summaries(test_batches, steps)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps)

        if self.validation_dataset is not None and (
                steps % self.cfg['training']['validation_steps'] == 0
                or steps % self.cfg['training']['total_steps'] == 0):
            with tf.profiler.experimental.Trace("Validate", step_num=steps):
                if self.swa_enabled:
                    self.calculate_swa_validations(steps)
                else:
                    self.calculate_test_validations(steps)

        # Save session and weights at end, and also optionally every 'checkpoint_steps'.
        if steps % self.cfg['training']['total_steps'] == 0 or (
                'checkpoint_steps' in self.cfg['training']
                and steps % self.cfg['training']['checkpoint_steps'] == 0):
            evaled_steps = steps.numpy()
            self.manager.save(checkpoint_number=evaled_steps)
            print("Model saved in file: {}".format(
                self.manager.latest_checkpoint))
            path = os.path.join(self.root_dir, self.cfg['name'])
            leela_path = path + "-" + str(evaled_steps)
            swa_path = path + "-swa-" + str(evaled_steps)
            self.net.pb.training_params.training_steps = evaled_steps
            self.save_leelaz_weights(leela_path)
            if self.swa_enabled:
                self.save_swa_weights(swa_path)

        if self.profiling_start_step is not None and (
                steps >= self.profiling_start_step +
                self.cfg['training'].get('profile_step_count', 0)
                or steps % self.cfg['training']['total_steps'] == 0):
            tf.profiler.experimental.stop()
            self.profiling_start_step = None

    def calculate_swa_summaries(self, test_batches, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print('swa', end=' ')
        self.calculate_test_summaries(test_batches, steps)
        self.test_writer = true_test_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y):
        outputs = self.model(x, training=False)
        policy = outputs #outputs[0]
        policy_loss = self.policy_loss_fn(y, policy)
        policy_accuracy = self.policy_accuracy_fn(y, policy)
        policy_entropy = self.policy_entropy_fn(y, policy)
        policy_ul = self.policy_uniform_loss_fn(y, policy)
        metrics = [
            policy_loss,
            policy_accuracy * 100,
            policy_entropy,
            policy_ul,
        ]
        return metrics

    @tf.function()
    def strategy_calculate_test_summaries_inner_loop(self, x, y):
        metrics = self.strategy.run(self.calculate_test_summaries_inner_loop,
                                    args=(x, y))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics

    def calculate_test_summaries(self, test_batches, steps):
        for metric in self.test_metrics:
            metric.reset()
        for _ in range(0, test_batches):
            x, y = next(self.test_iter)
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(x, y)
            else:
                metrics = self.calculate_test_summaries_inner_loop(x, y)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        self.net.pb.training_params.learning_rate = self.lr
        self.net.pb.training_params.policy_loss = self.test_metrics[0].get()
        # TODO store value and value accuracy in pb
        self.net.pb.training_params.accuracy = self.test_metrics[1].get()
        with self.test_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
            for w in self.model.weights:
                tf.summary.histogram(w.name, w, step=steps)
        self.test_writer.flush()

        print("step {},".format(steps), end='')
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                      metric.suffix),
                  end='')
        print()

    def calculate_swa_validations(self, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_validation_writer, self.validation_writer = self.validation_writer, self.swa_validation_writer
        print('swa', end=' ')
        self.calculate_test_validations(steps)
        self.validation_writer = true_validation_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def calculate_test_validations(self, steps):
        for metric in self.test_metrics:
            metric.reset()
        for (x, y) in self.validation_dataset:
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(x, y)
            else:
                metrics = self.calculate_test_summaries_inner_loop(x, y)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        with self.validation_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
        self.validation_writer.flush()

        print("step {}, validation:".format(steps), end='')
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                      metric.suffix),
                  end='')
        print(flush=True)

    @tf.function()
    def compute_update_ratio(self, before_weights, after_weights, steps):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [
            after - before
            for after, before in zip(after_weights, before_weights)
        ]
        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [
            tf.math.reduce_euclidean_norm(w) for w in before_weights
        ]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.))
                  for d, w, tensor in zip(delta_norms, weight_norms,
                                          self.model.weights)
                  if not 'moving' in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar('update_ratios/' + name, ratio, step=steps)
        # Filtering is hard, so just push infinities/NaNs to an unreasonably large value.
        ratios = [
            tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299,
                    lambda: 200.) for (_, r) in ratios
        ]
        tf.summary.histogram('update_ratios_log10',
                             tf.stack(ratios),
                             buckets=1000,
                             step=steps)

    def update_swa(self):
        num = self.swa_count.read_value()
        for (w, swa) in zip(self.model.weights, self.swa_weights):
            swa.assign(swa.read_value() * (num / (num + 1.)) + w.read_value() *
                       (1. / (num + 1.)))
        self.swa_count.assign(min(num + 1., self.swa_max_n))

    def save_swa_weights(self, filename):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        self.save_leelaz_weights(filename)
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def save_leelaz_weights(self, filename):
        numpy_weights = []
        for weight in self.model.weights:
            numpy_weights.append([weight.name, weight.numpy()])
        # self.net.fill_net_v2(numpy_weights) TODO: fix
        self.net.save_proto(filename)

    def batch_norm(self, input, name, scale=False):
        if self.renorm_enabled:
            clipping = {
                "rmin": 1.0 / self.renorm_max_r,
                "rmax": self.renorm_max_r,
                "dmax": self.renorm_max_d
            }
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                fused=False,
                center=True,
                scale=scale,
                renorm=True,
                renorm_clipping=clipping,
                renorm_momentum=self.renorm_momentum,
                name=name)(input)
        else:
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                center=True,
                scale=scale,
                virtual_batch_size=self.virtual_batch_size,
                name=name)(input)

    def squeeze_excitation(self, inputs, channels, name):
        assert channels % self.SE_ratio == 0

        pooled = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_first')(inputs)
        squeezed = tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
            tf.keras.layers.Dense(channels // self.SE_ratio,
                                  kernel_initializer='glorot_normal',
                                  kernel_regularizer=self.l2reg,
                                  name=name + '/se/dense1')(pooled))
        excited = tf.keras.layers.Dense(2 * channels,
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=self.l2reg,
                                        name=name + '/se/dense2')(squeezed)
        return ApplySqueezeExcitation()([inputs, excited])

    def conv_block(self,
                   inputs,
                   filter_size,
                   output_channels,
                   name,
                   bn_scale=False):
        conv = tf.keras.layers.Conv2D(output_channels,
                                      filter_size,
                                      use_bias=False,
                                      padding='same',
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      data_format='channels_first',
                                      name=name + '/conv2d')(inputs)
        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
            self.batch_norm(conv, name=name + '/bn', scale=bn_scale))

    def residual_block(self, inputs, channels, name):
        conv1 = tf.keras.layers.Conv2D(channels,
                                       3,
                                       use_bias=False,
                                       padding='same',
                                       kernel_initializer='glorot_normal',
                                       kernel_regularizer=self.l2reg,
                                       data_format='channels_first',
                                       name=name + '/1/conv2d')(inputs)
        out1 = tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
            self.batch_norm(conv1, name + '/1/bn', scale=False))
        conv2 = tf.keras.layers.Conv2D(channels,
                                       3,
                                       use_bias=False,
                                       padding='same',
                                       kernel_initializer='glorot_normal',
                                       kernel_regularizer=self.l2reg,
                                       data_format='channels_first',
                                       name=name + '/2/conv2d')(out1)

        out2 = self.squeeze_excitation(self.batch_norm(conv2,
                                                       name + '/2/bn',
                                                       scale=True),
                                       channels,
                                       name=name + '/se')
        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
            tf.keras.layers.add([inputs, out2]))

    @staticmethod
    def split_heads(inputs, batch_size: int, num_heads: int, depth: int):
        if num_heads < 2:
            return inputs
        reshaped = tf.reshape(inputs, (batch_size, 64, num_heads, depth))
        # (batch_size, num_heads, 64, depth)
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])

    def create_residual_body(self, inputs):
        flow = self.conv_block(inputs,
                               filter_size=3,
                               output_channels=self.RESIDUAL_FILTERS,
                               name='input',
                               bn_scale=True)
        for i in range(self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow,
                                       self.RESIDUAL_FILTERS,
                                       name='residual_{}'.format(i + 1))
        return flow

    def construct_net(self, inputs, name=''):

        flow = self.create_residual_body(inputs)

        # Policy head
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_CONVOLUTION:
            conv_pol = self.conv_block(flow,
                                       filter_size=3,
                                       output_channels=self.RESIDUAL_FILTERS,
                                       name='policy1')
            conv_pol = keras_cv.layers.DropBlock2D(0.1, block_size=4, data_format='channels_first')(conv_pol)
            conv_pol2 = tf.keras.layers.Conv2D(
                64,
                3,
                use_bias=True,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                bias_regularizer=self.l2reg,
                data_format='channels_first',
                name='policy')(conv_pol)
            h_fc1 = ApplyPolicyMap()(conv_pol2)
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_CLASSICAL:
            conv_pol = self.conv_block(flow,
                                       filter_size=1,
                                       output_channels=self.policy_channels,
                                       name='policy')
            h_conv_pol_flat = tf.keras.layers.Flatten()(conv_pol)
            h_fc1 = tf.keras.layers.Dense(1792,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          bias_regularizer=self.l2reg,
                                          name='policy/dense')(h_conv_pol_flat)
        else:
            raise ValueError("Unknown policy head type {}".format(
                self.POLICY_HEAD))

        outputs = [h_fc1]
        return outputs
