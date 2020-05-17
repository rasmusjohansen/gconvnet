from gconvnet.groups.group import Group
import numpy as np

class Dn(Group):
    """Dihedral group of order 2n."""
    
    def __init__(self,n):
        super(Dn,self).__init__()
        
        self.size = 2*n
        self.ndim = 1
        self.shape = (2*n,)
        self.identity = np.array([0],dtype=np.int32)
        
    
    def inv(self,x):
        x = self.clean_input(x)        
        
        rot_x = x[0] // 2
        ref_x = x[0] % 2
        
        if ref_x == 1:
            return self.clean_input(x[0])
        else:
            return self.clean_input(2 * (-rot_x) % self.size)
        
    def mul(self,x,y):
        x = self.clean_input(x)  
        y = self.clean_input(y)  
        
        rot_x = x // 2
        ref_x = x % 2
        rot_y = y // 2
        ref_y = y % 2
        
        if ref_x == 1:
            rot = rot_x - rot_y
        else:
            rot = rot_x + rot_y
        
        rot = rot % self.size
        ref = (ref_x + ref_y) % 2
        
        return self.clean_input(ref + 2 * rot)
    
    def __str__(self):
        return 'D' + str(self.size//2)