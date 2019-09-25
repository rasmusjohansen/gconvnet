import gconvnet.groups as groups
import numpy as np

class GMaxPool(layers.Layer):


    def __init__(self,
                 subgroup_inclusion,
                 name=None,
                 **kwargs):
        super(GMaxPool,self).__init__(
            name=name,
            **kwargs)

        self.H = groups.get_group(subgroup_inclusion)
        self.G = self.H.supergroup
        self.GmodH = groups.create_quotient_group(
            self.H)

    def build(self, input_shape):

        if input_shape.rank != 5:
            raise ValueError('Unexpected input_shape in GMaxPool')

        if input_shape[3] != self.G.size:
            raise ValueError('Unexpected G-dimension in GMaxPool')

        self.build_indices()

        super(GMaxPool, self).build(input_shape)

    def call(self, inputs):

        # Transforms inputs from
        # (batch,width,height,G.size,ch)
        # to
        # (batch,width,height,H.size, GmodH.size, ch)
        # so we can max reduce along H.
        
        inputs2 = tf.gather(inputs,
                            self.quotient_indices,
                            axis=3)

        inputs2 = tf.reduce_max(inputs2,
                                axis=3)

        return inputs2
        

    def build_indices(self):

        # quotient_indices represents a bijection:
        # H x (G/H) -> G
        # such that (h, gmodh) is sent to an element g
        # in the coset gmodh.
        
        self.quotient_indices = np.empty(shape=(self.H.size,
                                                self.GmodH.size),
                                         dtype=np.int32)

        for gmodh_idx in range(self.GmodH.size):
            g_idx = self.GmodH.GmodH_to_G[gmodh_idx]
            g = self.G(g_idx) 
            
            for h_idx in range(self.H.size):
                h = self.H(h_idx)

                gh = g * h.g_elem
                gh_idx = int(gh)

                self.quotient_indices[h_idx,
                                      gmodh_idx] = gh_idx

        
        
