from gconvnet.groups.group import Group
import numpy as np

class Cn(Group):
    """Cyclic group of order n."""
    def __init__(self,n):
        assert n >= 1
        
        super(Cn,self).__init__()
        
        self.size = n
        self.ndim = 1
        self.shape = (n,)
        self.identity = np.array([0],dtype=np.int32)
        self.gens = {1}
        
    def inv(self,x):
        x = self.clean_input(x)
        return self.clean_input( 
            (self.size - x[0]) % self.size )

    def mul(self,x,y):
        x = self.clean_input(x)
        y = self.clean_input(y)
        
        return self.clean_input(
            (x[0]+y[0]) % self.size)
    
    def __str__(self):
        return 'Z/' + str(self.size) + 'Z'