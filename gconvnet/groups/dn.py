class dn:
    """Dihedral group of order 2n."""
    
    def __init__(self,n):
        self.size = 2*n
        self.identity = 0
        
    
    def inv(self,x):
        rot_x = x // 2
        ref_x = x % 2
        
        if ref_x == 1:
            return x
        else:
            return 2 * (-rot_x) % self.size
        
    def mul(self,x,y):
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
        
        return ref + 2 * rot