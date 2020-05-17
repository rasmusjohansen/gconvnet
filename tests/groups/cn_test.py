from gconvnet.groups import Cn
from bruteforce_util import *
import numpy as np

def test_size():
    G1 = Cn(1)
    G2 = Cn(2)
    G3 = Cn(3)
    G4 = Cn(4)
    G5 = Cn(5)
    G6 = Cn(6)
    G7 = Cn(7)
    G8 = Cn(8)
    
    assert G1.size == 1
    assert G2.size == 2
    assert G3.size == 3
    assert G4.size == 4
    assert G5.size == 5
    assert G6.size == 6
    assert G7.size == 7
    assert G8.size == 8
    
def test_identity():
    G7 = Cn(7)
    
    for i in range(G7.size):
            assert G7.mul(G7.identity, i) == np.array([i],dtype=np.int32)
            assert G7.mul(i,G7.identity) == np.array([i],dtype=np.int32)
            
            
def test_failed():
    G5 = Cn(5)
    
    dtest_group_identity(G5)
    
    
    
    
    
    
    