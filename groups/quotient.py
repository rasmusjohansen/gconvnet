import numpy as np

def create_quotient_group(H,name=None):
    """Creates the quotient G/H from H"""

    G = H.supergroup
    
    if name == None and H.name != None and G.name != None:
        name = G.name + '/' + H.name
    
    _name = name


    assert G.size % H.size == 0
    
    class QuotientGroup:

        size = G.size // H.size
        name = _name
        subgroup = H
        supergroup = G

        def __init__(self,index):
            self.GmodH_index = index
            self.G_index = self.GmodH_to_G[index]
            self.G_elem = self.supergroup(self.G_index)

        def __mul__(self, other):
            G_product = self.G_elem * other.G_elem

            GmodH_index = self.G_to_GmodH[G_product]

            assert GmodH_index != -1

            return QuotientGroup(GmodH_index)

        def inverse(self):
            G_inverse = self.G_elem.inverse()
            GmodH_index = self.G_to_GmodH[G_inverse]
            
            assert GmodH_index != -1

            return QuotientGroup(GmodH_index)

        def __eq__(self,other):
            return (self.G_elem == other.G_elem)

        def __int__(self):
            return int(self.GmodH_index)

        def __str__(self):
            if self.subgroup.name:
                return str(self.G_elem) + H.name
            else:
                return '(' + str(self.G_elem) + ' coset)'

    QuotientGroup.G_to_GmodH = np.full((G.size),
                                       -1,
                                       dtype=np.int32)

    QuotientGroup.GmodH_to_G = np.full((QuotientGroup.size),
                                       -1,
                                       dtype=np.int32)
    
    coset_idx = 0            
    for g_idx in range(G.size):

        print(QuotientGroup.GmodH_to_G)
        
        g = G(g_idx)
        
        # If no index has been assigned to g_idx yet,
        # assign an index to the entire coset.
        if QuotientGroup.G_to_GmodH[g_idx] == -1:

            QuotientGroup.GmodH_to_G[coset_idx] = g_idx
            
            for h_idx in range(H.size):

                h = H(h_idx)
                gh = g * h.G_elem

                gh_idx = int(gh)

                # If index has already been assigned something
                # has gone wrong. Shouldn't be possible if H is
                # a normal subgroup of H.
                assert QuotientGroup.G_to_GmodH[gh_idx] == -1
                
                QuotientGroup.G_to_GmodH[gh_idx] = coset_idx
                
            coset_idx += 1

    return QuotientGroup
