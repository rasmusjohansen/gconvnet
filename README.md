# gconvnet
G-CNN layer implementation for Tensorflow/Keras.


## Representation of groups

A group is represented by a class. An element of the group is represented by an object of that class. At a minimum a group G should implement:

* G() should construct the identity element.

* G.__mul__ does group composition.

* G.inverse computes the inverse group element and returns it.

* G.__eq__ should test for equality of group elements.

* G.name String name of the group. Set to None if no name is provided.

Additionally groups should if possible implement the following:

* G.__str__ string representation of the group element.

For a group G acting on a convolutional layer we will additionally require:

* G is a finite group.

* G.size Integer representing the number of elements in the group.

* G.__int__ Returns an integer representation of the group element. This should
  provide a bijection between the group elements and the integers 0,1,...,G.size-1.

* If integer representation is passed to G.__init__ it should construct the corresponding
  group element. In particular we should have int(G(k)) == k for k in 0,1,...,G.size-1.

For lightweight groups consider using the utility functions:

* create_indexed_group_class for finite groups with integer representations.

* create_group_class for infinite groups.


### Sub- and quotient groups

Currently subgroups and quotientgroups are only implemented for finite groups with integer
representation.

A subgroup of G is a group H for which we have the following:

* H is a subgroup of G in the group theoretic sense.

* H.G_to_H is a G.size sized integer array translating G integer representations
  to H integer representations. Any G integer representation that does not
  correspond to an element of H should map to -1.

* H.H_to_G is an H.size sized integer array translating H integer representations
  to G integer representations.

* H.supergroup should be G

Note: What we call a subgroup might more correctly be called a subgroup inclusion. The distinction serves as a distraction for end-users so we stick to calling it a subgroup even though technically it is slightly more. 

A quotient group G/H is a group GmodH for which we have the following:

* GmodH.supergroup is a group G

* GmodH.subgroup is H, a normal subgroup of G (warning: it is up to the user to check
  that H is a normal subgroup. If H is not a normal subgroup of G no guarantees are
  provided regarding behavior.

* GmodH.G_to_GmodH is a G.size sized integer array translating G integer representations
  to H integer representations.

* GmodH.GmodH_to_G is a (G.size/H.size) sized integer array translating coset indices to
  a G index of some group element of G in the coset. Users should not depend on any particular
  rule regarding which element of the coset is chosen as the representative.

