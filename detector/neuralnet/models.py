"""models.py: Module that defines trainable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import param_keys
from utils import constants
from utils import errors
from .base_model import BaseModel
from . import networks
from . import net_ops


class WaveletBLSTM(BaseModel):
    """ Model that manages the implemented network."""
    def __init__(self, params, logdir='logs'):
        """Constructor.

        Feat and label shapes can be obtained from params for this model.
        """
        self.params = param_keys.default_params
        self.params.update(params)  # Overwrite defaults
        page_duration = self.params[param_keys.PAGE_DURATION]
        border_duration = self.params[param_keys.BORDER_DURATION]
        fs = self.params[param_keys.FS]
        signal_length = int(2*fs*(page_duration + border_duration))
        feat_shape = [signal_length]
        label_shape = feat_shape
        super().__init__(feat_shape, label_shape, params, logdir)

    def _map_fn(self, feat, label):
        """Random cropping.

        This method is used to preprocess features and labels of single
        examples with a random cropping
        """
        time_stride = self.params[param_keys.TIME_RESOLUTION_FACTOR]
        border_duration = self.params[param_keys.BORDER_DURATION]
        page_duration = self.params[param_keys.PAGE_DURATION]
        fs = self.params[param_keys.FS]
        border_size = int(fs * border_duration)
        crop_size = int(fs * page_duration) + 2 * border_size
        # Random crop
        label_cast = tf.cast(label, dtype=tf.float32)
        stack = tf.stack([feat, label_cast], axis=0)
        stack_crop = tf.random_crop(stack, [2, crop_size])
        feat = stack_crop[0, :]
        # Throw borders for labels, skipping steps
        label_cast = stack_crop[1, border_size:-border_size:time_stride]
        label = tf.cast(label_cast, dtype=tf.int32)
        return feat, label

    def _model_fn(self):
        """
        This method is used to evaluate the model with the inputs, and return
        logits and probabilities.
        """
        logits, probabilities = networks.wavelet_blstm_net(
            self.feats, self.params, self.training_ph)
        return logits, probabilities

    def _loss_fn(self):
        """
        This method is used to return the loss between the output of the
        model and the desired labels.
        """
        type_loss = self.params[param_keys.TYPE_LOSS]
        errors.check_valid_value(
            type_loss, 'type_loss',
            [constants.CROSS_ENTROPY_LOSS, constants.DICE_LOSS])

        if type_loss == constants.CROSS_ENTROPY_LOSS:
            loss = net_ops.cross_entropy_loss_fn(
                self.logits, self.labels, self.params[param_keys.CLASS_WEIGHTS])
        else:
            loss = net_ops.dice_loss_fn(self.probabilities[..., 1], self.labels)
        return loss

    def _optimizer_fn(self):
        """
        This method is used to define the operation train_step that performs
        one training iteration to optimize the loss, and the reset optimizer
        operation that resets the variables of the optimizer, if any.
        """
        type_optimizer = self.params[param_keys.TYPE_OPTIMIZER]
        errors.check_valid_value(
            type_optimizer, 'type_optimizer',
            [constants.ADAM_OPTIMIZER, constants.SGD_OPTIMIZER])

        if type_optimizer == constants.ADAM_OPTIMIZER:
            train_step, reset_optimizer_op = net_ops.adam_optimizer_fn(
                self.loss, self.learning_rate,
                self.params[param_keys.CLIP_GRADIENTS],
                self.params[param_keys.CLIP_NORM])
        else:
            train_step, reset_optimizer_op = net_ops.sgd_optimizer_fn(
                self.loss, self.learning_rate,
                self.params[param_keys.MOMENTUM],
                self.params[param_keys.CLIP_GRADIENTS],
                self.params[param_keys.CLIP_NORM])
        return train_step, reset_optimizer_op

    def _metrics_fn(self):
        """This method is used to compute several useful metrics based on the
        model output and the desired labels. The metrics are stored in a metrics
        dictionary that is returned.
        """
        # TODO: metrics implementation for training loop.
        labels = self.labels
        probabilities = self.probabilities
        metrics_dict = {}
        return metrics_dict
