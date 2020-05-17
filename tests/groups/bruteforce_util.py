import numpy as np

def dtest_group_identity(G):
    def replace_inf(x,new_inf):
        if x == 0:
            return new_inf
        else:
            return x
        
    
    max_size = 50
    
    shape = tuple([replace_inf(x,max_size) for x in G.shape])
    
    for multi_index in np.ndindex(shape):
        assert G.clean_input(multi_index) == \
            G.mul(multi_index, G.identity)
        assert G.clean_input(multi_index) == \
            G.mul(G.identity,multi_index)
    