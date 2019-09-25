import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn,nn_ops
import numpy as np
import gconvnet.groups as groups

class GConv2D(layers.Layer):
    """2D G-convnet layer.

    Important shapes:
        raw filterbank: (kw,kh,G_in.size,ch_in,ch_out)

        filterbank: (kw,kh,G_in.size,ch_in,G_out.size,ch_out)

        filter indices: (kw,kh,G_in.size,G_out.size).

        output: (?,w,h,G_out.size,ch_out)"""
    
    def __init__(self, 
                 filters, 
                 kernel_size,
                 G,
                 G_action,
                 G_in = None,
                 G_action_on_G_in = None,                 
                 strides = None,
                 activation = None,
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer='zeros',
                 trainable=True,
                 name=None,
                **kwargs):
        """Initializes an equivariant convolutional layer.

        Arguments:
            filters: The number of channels to create for this layer.

            kernel_size: 2d tuple of ints indicating the size of each filter/kernel.
 
            G: The group which acts on the output layer. If it is a string, then the
            group will be looked up using groups.get_group().

            G_action: The action of G on Z^2. If string, it will be looked up using
            using groups.get_action()

            G_in: The group acting on the input layer. If None, then it is assumed to
            be either the trivial group or G.

            G_action_on_G_in: The action of G on G_in. If G_in = G or G_in is trivial,
            then it can be left as None.

            strides: Not implemented yet.

            activation: The activation function applied after the G-convolution. If
            string, then it is looked up using keras.activations.get

            use_bias: Whether to include a bias weight.

            kernel_initializer: How to initialize the filter/kernel.

            bias_initializer: How to initialize the bias matrix (if use_bias)."""
        super(GConv2D,self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        
        if isinstance(kernel_size,int):
            raise NotImplementedError('Kernel size extrapolation from int not implemented.' +
                                      'Please provide a complete tuple.')
        if strides != None:
            raise NotImplementedError('Non-trivial strides not implemented.')
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,kernel_size)
        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 2:
                raise ValueError('Kernel size expected dimension 2, but got dimension ' + str(len(kernel_size)))
            self.kernel_size = kernel_size
        else:
            raise ValueError('Kernel size type not recognized')

        self.kw = self.kernel_size[0]
        self.kh = self.kernel_size[1]
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.ch_out = filters
        self.activation = tf.keras.activations.get(activation)
        self.G_out = groups.get_group(G)
        self.G_out_action = groups.get_action(G_action)
        self.G_in = groups.get_group(G_in)
        self.G_action_on_G_in = groups.get_action(G_action_on_G_in)
        
        
       # self.filterbank = self.add_weight(shape=(kernel_size[0],kernel_size[1],num_channels),
        #                initializer='random_normal',
       #                 trainable=True)
      #  self.bias = self.add_weight(shape=(output_shape[0],output_shape[1],output_shape[2]))
        
    def build(self,input_shape):

        # If input feature map has a G-action, then input shape should
        # be (?, Width, Height, G_in.size, Channels_in)
        # If G_in.size is missing (i.e. rank=4), then there is no G-action.
        # If G_in.size = 1, then there is no G_in action.

        if input_shape.rank == 4:
            raise NotImplementedError('Automatic rank extension not implemented')

        if input_shape.rank != 5:
            raise ValueError('Invalid input shape')

        self.spatial_w = int(input_shape[1])
        self.spatial_h = int(input_shape[2])
        self.ch_in = int(input_shape[4])

        G_in_size = input_shape[3]

        # if (G_in is trivial)
        if self.G_in == None and G_in_size == 1:
            self.G_in = groups.c1
            self.G_action_on_G_in = groups.trivial_action

        # if (G_in = G_out)
        elif self.G_in == None and G_in_size == self.G_out.size:
            self.G_in = self.G_out
            self.G_action_on_G_in = groups.multiplication_action

        elif self.G_in == None:

            raise ValueError('Unable to infer appropriate G_in')

        elif self.G_action_on_G_in == None:

            raise ValueError('G_in provided, but no corresponding action of G_out'
                             'provided.')

        assert self.G_in != None
        assert self.G_action_on_G_in != None
        assert self.G_in.size == G_in_size
        
            
        self.raw_filterbank_shape = (self.kw,
                                     self.kh,
                                     self.G_in.size,
                                     self.ch_in,
                                     self.ch_out)

        self.filter_indices_shape = (self.kw,
                                     self.kh,
                                     self.G_in.size,
                                     self.G_out.size)

        self.filterbank_shape = (self.kw,
                                 self.kh,
                                 self.G_in.size,
                                 self.ch_in,
                                 self.G_out.size,
                                 self.ch_out)
        
        assert self.raw_filterbank_shape[:2] == self.kernel_size
                    
        self.raw_filterbank = self.add_weight(
            name='raw_filterbank',
            shape=self.raw_filterbank_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype)
        
        if self.use_bias:
            raise NotImplementedError('Bias not implemented')
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters_raw,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self.create_filter_indices()

        #raw filterbank shape: (kw,kh,G_in,ch_in,ch_out)
        #indices shape: (kw,kh,G_in,G_out)
        
        self.raw_filterbank2 = tf.reshape(
            self.raw_filterbank,
            [self.kw * self.kh * self.G_in.size,
             self.ch_in,
             self.ch_out])

        # filterbank temporary shape: (kw,kh,G_in,G_out,ch_in,ch_out)
        self.filterbank = tf.gather(self.raw_filterbank2,
                                    self.filter_indices)

        # new filterbank shape: (kw,kh,G_in,ch_in,G_out,ch_out)
        self.filterbank = tf.transpose(self.filterbank, [0,1,2,4,3,5])

        super(GConv2D,self).build(input_shape)
        
    def call(self,inputs):
        
        inputs_reshaped = tf.reshape(
            inputs,
            (-1,
             inputs.shape[1], inputs.shape[2],
             self.G_in.size * self.ch_in))
        filterbank_reshaped = tf.reshape(
            self.filterbank,
            (self.kw,
             self.kh,
             self.G_in.size * self.ch_in,
             self.G_out.size * self.ch_out))
        
        outputs = nn_ops.Convolution(
            inputs_reshaped.shape,
            filter_shape=filterbank_reshaped.shape,
            padding='VALID')(inputs_reshaped, filterbank_reshaped)

        outputs = tf.reshape(outputs,
                             (-1,
                              outputs.shape[1], outputs.shape[2],
                              self.G_out.size,
                              self.ch_out))
        
      #  outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')
        
        return self.activation(outputs)
        
    def compute_output_shape(self,input_shape):
        raise NotImplementedError('xyz')



    def filter_index_build(self):
        
        self.w_is_odd = (self.kw % 2 == 1)
        self.h_is_odd = (self.kh % 2 == 1)

        if self.w_is_odd:
            self.kw_min = -(self.kw - 1) // 2
            self.kw_max = (self.kw -1) // 2
            self.kw_step = 1
        else:
            self.kw_min = -self.kw + 1
            self.kw_max =  self.kw - 1
            self.kw_step = 2

        if self.h_is_odd:
            self.kh_min = -(self.kh - 1) // 2
            self.kh_max =  (self.kh - 1) // 2
            self.kh_step = 1
        else:
            self.kh_min = -self.kh + 1
            self.kh_max =  self.kh - 1
            self.kh_step = 2

    
    def filter_index_to_action(self,p):
        """Converts an index into a kw x kh matrix to
        the corresponding kernel translation action.

        1d examples:
            rank 1: 0 -> 0
            rank 2: 0,1 -> -1, 1
            rank 3: 0,1,2 -> -1,0,1
            rank 4: 0,1,2,3 -> -3,-1,1,3
            rank 5: 0,1,2,3,4 -> -2,-1,0,1,2."""
        i,j = p[0],p[1]

        x =  self.kw_step * i + self.kw_min
        y = -self.kh_step * j + self.kh_max

        return groups.z2((x,y))
        
    def filter_action_to_index(self,t):
        x = t.data[0]
        y = t.data[1]
        
        i = (x - self.kw_min) // self.kw_step
        j = (self.kh_max - y) // self.kh_step
    
        return (i,j)
        
    
    def create_filter_indices(self):
        self.filter_index_build()

        # Final shape: (kw, kh, G_in.size, G.size)
        self.filter_indices = np.empty(
            shape=self.filter_indices_shape,
            dtype = np.int32)

        # Just iterate through (kw,kh,G_in,G) and calculate
        # the right index.

        # Optimization: This code is not particularly efficient. It could
        # easily be parallelized in C++ or as a GPU routine. However in
        # most cases kw * kh * G_in.size * G.size is so small that it doesn't
        # matter.
        
        for g_out_idx in range(self.G_out.size):
            g_out = self.G_out(g_out_idx)
            g_out_inv = g_out.inverse()

            for g_in_idx in range(self.G_in.size):
                g_in = self.G_in(g_in_idx)
                new_g_in = self.G_action_on_G_in(g_out_inv, g_in)

                for i in range(self.kw):
                    for j in range(self.kh):
                        old_translation = self.filter_index_to_action((i,j))

                        new_translation = self.G_out_action(g_out_inv,
                                                            old_translation)

                        new_i,new_j = self.filter_action_to_index(new_translation)

                        idx = int(new_g_in) + \
                            new_j * self.G_in.size + \
                            new_i * self.G_in.size * self.kh
                        self.filter_indices[i,j,g_in_idx,g_out_idx] = idx


                        
                
