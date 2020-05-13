import numpy as np

class Zn:    
    def __init__(self,n):
        self.size = 0
        self.ndim = n
        self.identity = np.zeros((n,))        
    
    def inv(self,x):
        assert isinstance(x,np.ndarray)
        assert x.ndim == 1
        assert len(x)==self.ndim
        
        return -x
        
    def mul(self, x, y):
        assert isinstance(x,np.ndarray)
        assert x.ndim == 1
        assert len(x)==self.ndim
        assert isinstance(y,np.ndarray)
        assert y.ndim == 1
        assert len(y)==self.ndim
        
        return x+y