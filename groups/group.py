from .get import register_group, register_action

def create_group_class(
        constructor_fn,
        composition_fn,
        inverse_fn,
        identity,
        eq_fn = None,
        str_fn = None,
        name = None):
    """Creates group class from specified functions.

    Arguments:
        constructor_fn(*args): Constructs a group object based on *args.
        Should return the identity if called with no arguments.

        composition_fn(x,y): Computes the composition of x and y.

        inverse_fn(x): Computes the inverse of group element x.

        identity: The identity group element.

        eq_fn(x,y): Tests group elements x,y for equality. If None, then ==
        is used for comparison.

        str_fn(x): Converts group element x to a string. __str__ will raise
        NotImplementedError if this is not provided.

        name: Descriptive name of the group.

    Returns:
        Group class constructed from the given functions."""

    id_ = identity
    name_ = name
    class _Group:
        identity = id_
        name = name_
        
        def __init__(self,*args):
            self.data = constructor_fn(*args)

        def inverse(self):
            data = inverse_fn(self.data)
            return self.__class__(data)

        def __mul__(self,rhs):
            data=composition_fn(self.data,rhs.data)
            return self.__class__(data)

        def __eq__(self,rhs):
            if eq_fn is None:
                return self.data == rhs.data
            else:
                return eq_fn(self.data,rhs.data)

        def __str__(self):
            if str_fn:
                return str_fn(self.data)
            raise NotImplementedError('Group ' + self.name + ' has no __str__ function')

    return _Group

def create_indexed_group_class(
        composition_fn,
        inverse_fn,
        size,
        identity=0,
        str_fn = None,
        name=None):
    """Creates an indexed group class from specified functions.

    An indexed group of size n assumes that the group elements are encoded as:
    0,1,2,...,n-1.

    For infinite group the elements are encoded as:
    0,1,2,...

    It is assumed that each index corresponds uniquely to precisely one group
    element.

    Arguments:
        composition_fn(x,y): Computes the composition of x and y.

        inverse_fn(x): Computes the inverse of group element x.

        size: Size of the group. None for infinite group.

        identity: The identity group element. Default: 0

        str_fn(x): Converts group element x to a string. If None,
        then the index will just be converted to a string.

        name: Descriptive name of the group.

    Returns:
        Indexed group class constructed from the given functions."""

    def constructor(*args):
        if len(args) == 0:
            return 0
        else:
            return args[0]

    if str_fn is None:
        str_fn = lambda i: str(i)

    eq_fn = lambda x,y: x==y

    _Base = create_group_class(
        constructor,
        composition_fn,
        inverse_fn,
        identity,
        eq_fn,
        str_fn,
        name)

    _size = size
    
    class _Group(_Base):
        size = _size
        
        def __init__(self,index = 0):
            super(_Group,self).__init__(index)

        def __int__(self):
            return self.data

    return _Group
    

        
def create_cn_group_class(n,name=None):
    """Creates indexed group class representing the order n
    cyclic group C_n."""

    if name==None:
        name = 'C_' + str(n)

    return create_indexed_group_class(
        lambda x,y: (x+y)%n,
        lambda x: (n-x)%n,
        n,
        identity=0,
        str_fn=lambda i:str(i) +'(mod ' + str(n) + ')',
        name=name)

c1 = create_cn_group_class(1,name='Trivial group')
c2 = create_cn_group_class(2)
c4 = create_cn_group_class(4)

register_group(c1,'trivial')
register_group(c2,'c2')
register_group(c4,'c4')

def z2_constructor(*args):
    num_args = len(args)

    if num_args == 0:
        return (0,0)
    
    elif num_args == 1:
        if not isinstance(args[0],tuple):
            raise ValueError()
        if len(args[0]) != 2:
            raise ValueError()
        return args[0]
    
    elif num_args == 2:
        return (args[0],args[1])

    else:
        raise ValueError()

z2 = create_group_class(
    z2_constructor,
    lambda p1,p2: (p1[0]+p2[0], p1[1] + p2[1]),
    lambda p: (-p[0], -p[1]),
    (0,0),
    name='Z2')    

register_group(z2,'z2')

def trivial_action(g,x):
    return x

def action_ccw90(g, p):
    g_int = int(g)
    assert g_int >= 0
    assert g_int < 4

    x,y = p.data[0], p.data[1]
    
    for _ in range(g_int):
        x,y = -y,x

    return p.__class__((x,y))

def multiplication_action(g1,g2):
    return g1 * g2

register_action(trivial_action,'trivial')
register_action(action_ccw90,'rot90')
register_action(multiplication_action,'selfaction')
