import gconvnet.groups as groups

class GMaxPool(layers.Layer):


    def __init__(self,
                 H,
                 GmodH,
                 assemblymap,
                 name=None,
                 **kwargs):
        super(GMaxPool,self).__init__(
            name=name,
            **kwargs)
        
        self.H = groups.get_group(H)
        self.GmodH = groups.get_group(GmodH)

        self.assemblymap = assemblymap

    def build(self, input_shape):
        raise NotImplementedError('')
        
