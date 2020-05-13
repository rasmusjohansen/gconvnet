import tensorflow as tf
import numpy as np

class EquivariantLayer(tf.keras.layers.Layer):
    def __init__(self,
                input_space,
                output_space,
                filter_sub,
                embedding,
                weight_partition_func,
                padding_value):
        super(EquivariantLayer,self).__init__()
        
        self.input_space = input_space
        self.output_space = output_space
        self.filter_sub = filter_sub
        self.embedding = embedding
        self.weight_partition_func = weight_partition_function
        self.padding_value = padding_value
        
    def build(self,input_shape):
        self.input_size = functools.reduce(lambda x,y: x*y, self.input_space.shape)
        self.output_size = functools.reduce(lambda x,y: x*y, self.output_space.shape)
        
        # weight_indices is an input_shape x output_shape tensor that maps
        # (input_coord,output_coord) to the weight index for connecting input input_coord
        # to output output_coord.
        # 0 is reserved for a missing connection.
        # all positive integers correspond to a unique weight.
        # weight sharing is accomplished by using the same index for several connections.
        self.weight_indices = np.zeros( 
            shape=self.input_space.shape + self.output_space.shape, 
            dtype=np.int32 )
        
        weight_dict = {}
        
        for g in self.filter_sub:
            for output_coord in np.ndindex(self.output_space.shape):
                weight_class = self.weight_partition_func(g,output_coord)
                
                if weight_class in weight_dict:
                    weight_index = weight_dict[weight_class]
                else:
                    weight_index = len(weight_dict)+1
                    weight_dict[weight_class] = weight_index
                
                # Input is calculated by first mapping to the input space,
                # then applying the group action.
                input_coord = self.input_space.action(
                    g,
                    self.embedding(output_coord))    
                
                # TODO: Add proper support for padded coordinates.
                if not self.input_space.valid(input_coord):                
                    self.weight_indices[input_coord + output_coord] = weight_index
                        
                    
        # TODO: Add initializer option.
        # TODO: Add bias option. Make sure bias works correct when parameter sharing.
        self.w = self.add_weight(shape=(len(weight_dict),),
                                 initializer='random_normal',
                                 trainable=True)
        
        
    def call(self,inputs):
        # w_plus = [0] + weights
        self.w_plus = tf.concat(
            [tf.constant(np.array([0],dtype=np.float32)),
             self.w],
            axis=0)
            
        self.weight_matrix = tf.gather(
            self.w_plus, 
            tf.reshape(self.weight_indices,
                      shape=(self.input_size, self.output_size)))
                
        flat_inputs = tf.reshape(inputs, 
                                 shape=(-1,self.input_size))
                
        
        flat_output = tf.matmul(
            flat_inputs,
            self.weight_matrix)
                
        return tf.reshape(flat_output, 
                          shape=(-1,) + self.output_space.shape)   