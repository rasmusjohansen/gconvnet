import numpy as np


def create_inclusion(G,H_indices,name=None):
    """Creates a subgroup of G based on H_indices along with an
    inclusion."""

    _name = name

    class Subgroup:

        size = len(H_indices)
        name = _name
        supergroup = G
        
        def __init__(self, index):
            self.G_index = self.H_to_G[index]
            self.G_elem = G(self.G_index)

        def __mul__(self, other):
            G_product = self.G_elem * other.G_elem

            H_idx = self.G_to_H[int(G_product)]
            if H_idx == -1:
                raise ValueError('Subgroup not closed under composition.')
            
            return Subgroup(self.G_to_H[H_idx])
        
            return Subgroup(self.G_to_H[int(G_product)])

        def inverse(self):
            G_inverse = self.G_elem.inverse()

            H_idx = self.G_to_H[int(G_inverse)]
            if H_idx == -1:
                raise ValueError('Subgroup not closed under inverses.')
            
            return Subgroup(H_idx)

        def __eq__(self,other):
            return (self.G_elem == other.G_elem)

        def __int__(self):
            H_idx = self.G_to_H[self.G_index]

            assert H_idx != -1
            
            return H_idx 
        
        def __str__(self):
            return str(self.G_elem)
    
    Subgroup.H_to_G = np.copy(H_indices, dtype = np.int32)

    Subgroup.G_to_H = np.full(shape=(G.size), -1, dtype = np.int32)
    
    for h_idx,g_idx in enumerate(H_indices):
        Subgroup.G_to_H[g_idx] = h_idx

    return Subgroup
        
            
            
