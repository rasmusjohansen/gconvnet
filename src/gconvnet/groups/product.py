from gconvnet.groups.group import Group
import numpy as np


class GroupProduct(Group):
    def __init__(self,G1,G2):
        super(GroupProduct,self).__init__()
        
        self.G1 = G1
        self.G2 = G2
        
        self.size = G1.size * G2.size
        self.shape = G1.shape + G2.shape
        self.ndim = self.G1.ndim + self.G2.ndim
        self.identity = np.concatenate([
            G1.identity, G2.identity
        ])
        
    def inv(self,x):
        x1 = x[:self.G1.ndim]
        x2 = x[self.G1.ndim:]
        
        return np.concatenate([self.G1.inv(x1), self.G2.inv(x2)])
        
    def mul(self,x,y):
        x1 = x[:self.G1.ndim]
        x2 = x[self.G1.ndim:]
        y1 = y[:self.G1.ndim]
        y2 = y[self.G1.ndim:]
        
        z1 = self.G1.mul(x1,y1)
        z2 = self.G2.mul(x2,y2)
        
        return np.concatenate([z1,z2])
    
    def __str__(self):
        return str(self.G1) + ' x ' + str(self.G2)