import numpy as np

class Group:
    def __init__(self):
        pass
    
    def clean_input(self,x):
        if self.ndim == 1:
            if isinstance(x,int) or isinstance(x,np.int32):
                x = np.array([x],dtype=np.int32)

        else:
            if isinstance(x,list) or isinstance(x,tuple):
                x = np.array(x,dtype=np.int32)
               
        assert isinstance(x,np.ndarray)
        assert x.ndim == 1
        assert len(x)==self.ndim
        
        return x     
    
    def inv(self,x):
        raise NotImplementedError()

    def mul(self,x,y):
        raise NotImplementedError()
    
    def __str__(self):
        raise NotImplementedError()
