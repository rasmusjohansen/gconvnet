# gconvnet
G-CNN layer implementation for Tensorflow/Keras. Based on [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576).

Use `gconvnet.GConv2D` to add a G-convolutional layer.

Use `gconvnet.GMaxPool` to add max-pooling across the group action.

Use `gconvnet.SpatialMaxPool` to add max-pooling across the underlying Z^2 grid.

## Representation of groups

A finite group with n elements will have its n elements represented by 0,1,...,n-1 with 0 being identity. If this is inconvenient
it is simple to write a wrapper that converts a different type of group specification to this.

A finite group with n elements will be represented by a class instance with the following properties:

- size: positive integer indicating size of group

- compose(x,y): Compute xy. Stateless.

- inverse(x): Compute x^(-1). Stateless.

