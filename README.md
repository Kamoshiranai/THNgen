# Tensor Hyper Network generator
Code to generate connected random Tensor Networks (TNs) and Tensor Hyper Networks (THNs). 
The output is in the form of einsum expressions (see [this blog post](https://rockt.github.io/2018/04/30/einsum) for an introduction to einsum and [numpy's implementation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) of einsum, which has a simple interface).

# Installation
#TODO

# Usage
#TODO

## Relation to opt_einsum.helpers.rand_equation
The python package [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/path_finding.html) has the function [opt_einsum.helpers.rand_equation](https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/helpers.py) which generates random TNs in einsum syntax. Those TNs are usually not connected (meaning they can be split into two or more independent TNs) and do not contain hyper edges, self edges, etc.
Our function random_tensor_network is an adaptation of this function and generates connected random TNs using the same input syntax.
Additionally, our function random_tensor_hyper_network generates connected random THNs which can be customized to fit all cases that can occur in an einsum expression (to the best of our knowledge).
