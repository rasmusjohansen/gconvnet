from gconvnet.groups import create_indexed_group_class
from gconvnet.groups import register_action, register_group
from gconvnet.groups import create_semidirect_product
from gconvnet.groups import action_ccw90
from gconvnet.groups import semidirect_product_action

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

def invert_action(g,x):
    g_idx = int(g)

    assert g_idx == 0 or g_idx == 1
    
    if g_idx == 0:
        return x
    else:
        return x.inverse()

    

c1 = create_cn_group_class(1,name='Trivial group')
c2 = create_cn_group_class(2,name='c2')
c4 = create_cn_group_class(4,name='c4')

register_group(c1,'trivial')
register_group(c2,'c2')
register_group(c4,'c4')
register_action(invert_action,'invert')

d4 = create_semidirect_product(c4,c2,invert_action)
d4_on_z2_action = semidirect_product_action(action_ccw90, invert_action)

register_group(d4,'d4')
register_action(d4_on_z2_action, 'd4')
