import gconvnet.groups as groups

def create_semidirect_product(
        N,
        H,
        action,
        name = None):
    """Constructs Group class representing the outer semidirect product of
    N with H via an action of H on N.

    Elements of the semidirect product can be represented as pairs (n,h) with
    n in N and h in H. Composition is defined by:
    
    (n1,h1) * (n2,h2) = (n1 * (h1 . n2), h1h2)
    
    Examples:
        - N = Z^2, H = Z/4Z, action = 90 degree rotations.
        Then the semiproduct is all symmetries of Z^2 composed of translations
        and 90 degree rotations. Also known as p4.
       
        - N = Z/nZ, H = Z/2Z, action = inversion.
        Result is dihedral group D_2n."""

    N_ = N
    H_ = H
    action_ = action
    name_ = name
    
    class _SemidirectGroup:
        N = N_
        H = H_
        action = action_
        name = name_
        
        size = N.size * H.size

        def NH_to_index(self,n,h):
            return int(n) + self.N.size * int(h)
        
        def __init__(self,
                     index=0):
            super(_SemidirectGroup,self).__init__()

            
            N_index = (index % self.N.size)
            H_index = (index-N_index) // self.N.size

            self.N_element = self.N(N_index)
            self.H_element = self.H(H_index)


        def __int__(self):
            N_index = int(self.N_element)
            H_index = int(self.H_element)

            return N_index + self.N.size * H_index

        def __mul__(self,other):
            N_product = self.N_element * \
                self.action.__func__(self.H_element, other.N_element)

            H_product = self.H_element * other.H_element

            
            return _SemidirectGroup(self.NH_to_index(N_product, \
                                                     H_product))
                     
            
        def inverse(self):

            # Semidirect product:
            # (n1,h1) * (n2,h2) = (n1 * (h1 . n2), h1h2)
            # If (n1,h1) is inverse of (n2,h2) we get:
            # h1h2 = 1 ===> h1 = (h2)^-1
            # n1 * (h1.n2) = 1 ===> n1 = (h1 . n2)^-1
            #
            # Here:
            # H_inverse = h1, N_inverse = n1
            # self.H_element = h2, self.N_element=n2
            
            H_inverse = self.H_element.inverse()

            hinv_n = self.action.__func__(H_inverse,self.N_element)
            N_inverse = hinv_n.inverse()

            ret_index = self.NH_to_index(N_inverse,H_inverse)
            return _SemidirectGroup(ret_index)
        

        def set_identity(self):
            self.H_element.set_identity()
            self.N_element.set_identity()
    
        def __eq__(self,other):
            N_check = (self.N_element == other.N_element)
            H_check = (self.H_element == other.H_element)

            return (N_check and H_check)

        def __str__(self):
            return '(' + str(self.N_element) + ', ' \
                + str(self.H_element) + ')'
        
            
    return _SemidirectGroup


def semidirect_product_action(
        N_action,
        H_action):
    
    def action(nh, x):
        n = nh.N_element
        h = nh.H_element

        hx = H_action(h,x)
        nhx = N_action(n,hx)

        return nhx

    return action
