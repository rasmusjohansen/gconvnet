import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

class ForgetAction(layers.Layer):
    """Layer which forgets the group action of a G-feature map. Useful before feeding the data to a layer incompatible with a 2d channel structure. In particular should be used before spatial max pooling."""

    def __init__(self,
                 name=None,
                 **kwargs):
        super(ForgetAction,self).__init__(
            name=name,
            **kwargs)

    def build(self,input_shape):
        super(ForgetAction,self).build(input_shape)

    def call(self,inputs):
        G_size = inputs.shape[-2]
        ch = inputs.shape[-1]

        new_ch = int(G_size * ch)

        outputs = tf.reshape(inputs,
                             inputs.shape[:-2].as_list() + [new_ch])

        return outputs

class RememberAction(layers.Layer):
    """Layer which remembers group action forgotten by ForgetAction"""

    def __init__(self,
                 G,
                 name=None,
                 **kwargs):
        self.G = G
        super(RememberAction,self).__init__(
            name=name,
            **kwargs)

    def build(self,input_shape):
        super(RememberAction,self).build(input_shape)

    def call(self,inputs):
        G_size = self.G.size
        ch = int(inputs.shape[-1]) // G_size

        outputs = tf.reshape(inputs,
                             inputs.shape[:-1].as_list() + [G_size,ch])

        return outputs
