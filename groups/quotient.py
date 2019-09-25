

def create_quotient_group(H,create_splitting=True,name=None):
    """Creates the quotient G/H from H"""

    _name = name
    G = H.supergroup
    
    class QuotientGroup:

        size = G.size / H.size
        name = _name
        subgroup = H
        supergroup = G

        def __init__(self,index):
            
        
        
