# Tensor Hyper Network generator
Code to generate connected random Tensor Networks (TNs) and Tensor Hyper Networks (THNs). 
The output is in the form of einsum expressions (see [this blog post](https://rockt.github.io/2018/04/30/einsum) for an introduction to einsum and [numpy's implementation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) of einsum, which has a simple interface).

# Installation
The only package necessary to run this code is numpy >= 1.17.0. It can be installed with e.g.  
conda create -n THNgen python numpy  
If you are using an older numpy version you will have to change the random number generation, the rest should still work.  
  
# Usage
## Generating Tensor Networks
Random Tensor Networks can be generated with the function generators.random_tensor_network (which has the same syntax as opt_einsum_helpers.rand_equation). 
  
Here is an example:  
  
eq, shapes, size_dict = random_tensor_network(  
    number_of_tensors = 10,  
    regularity = 3.5,  
    number_of_output_indices = 5,   
    min_axis_size = 2,  
    max_axis_size = 4,  
    return_size_dict = True,   
    global_dim = False,  
    seed = 12345  
)  
  
eq  
'gafoj,mpab,uhlbcdn,cqlipe,drstk,ve,fk,ongmq,hj,i->sturv'  
  
shapes  
[(3, 4, 4, 2, 3), (3, 2, 4, 2), (4, 4, 2, 2, 4, 2, 3), (4, 2, 2, 4, 2, 2), (2, 4, 3, 4, 4), (2, 2), (4, 4), (2, 3, 3, 3, 2), (4, 3), (4,)]  
  
size_dict  
{'a': 4, 'b': 2, 'c': 4, 'd': 2, 'e': 2, 'f': 4, 'g': 3, 'h': 4, 'i': 4, 'j': 3, 'k': 4, 'l': 2, 'm': 3, 'n': 3, 'o': 2, 'p': 2, 'q': 2, 'r': 4, 's': 3, 't': 4, 'u': 4, 'v': 2}  

## Generating Tensor Hyper Networks
Usual random Tensor Hyper Networks (basically Tensor Networks with hyper edges) can be generated with the function generators.random_tensor_hyper_network.  
  
Here is an example:  
  
eq, shapes, size_dict = random_tensor_hyper_network(  
    number_of_tensors = 10  
    regularity = 2.5  
    max_tensor_order = 10  
    max_edge_order = 5  
    number_of_output_indices = 5  
    min_axis_size = 2  
    max_axis_size = 4  
    return_size_dict = True,  
    seed = 12345  
)  
eq  
'bdca,abhcdg,cbmd,cfd,ed,e,figj,gl,h,nik->jnmkl'  
  
shapes  
[(2, 2, 2, 2), (2, 2, 4, 2, 2, 3), (2, 2, 4, 2), (2, 2, 2), (2, 2), (2,), (2, 4, 3, 3), (3, 2), (4,), (3, 4, 3)]  
  
size_dict  
{'a': 2, 'b': 2, 'c': 2, 'd': 2, 'e': 2, 'f': 2, 'g': 3, 'h': 4, 'i': 4, 'j': 3, 'k': 3, 'l': 2, 'm': 4, 'n': 3}  

It is also possible to generate more complicated structures, such as: self edges (of higher degree), output indices of higher degree, single summation indices, hyper edges with diagonals, output edges with diagonals.  

Here is an example:  
  
eq, shapes = random_tensor_hyper_network(  
    number_of_tensors = 10,   
    regularity = 3.0,  
    max_tensor_order = 10,  
    max_edge_order = 3,  
    diagonals_in_hyper_edges = True,  
    number_of_output_indices = 5,  
    max_output_index_order = 3,  
    diagonals_in_output_indices = True,  
    number_of_self_edges = 4,  
    max_self_edge_order = 3,  
    number_of_single_summation_indices = 3,  
    global_dim = True,  
    min_axis_size = 2,  
    max_axis_size = 4,  
    seed = 12345  
)  
eq  
'cabxk,gkegax,wldxbrb,ctoxdfo,xvdlv,weehx,nfnkx,spgpixqu,xjimhm,ijx->uvwtx'  
  
shapes  
[(3, 2, 4, 3, 2), (2, 2, 3, 2, 2, 3), (4, 4, 3, 3, 4, 3, 4), (3, 4, 3, 3, 3, 3, 3), (3, 3, 3, 4, 3), (4, 3, 3, 2, 3), (4, 3, 4, 2, 3), (3, 3, 2, 3, 2, 3, 2, 2), (3, 4, 2, 2, 2, 2), (2, 4, 3)]  

## Relation to opt_einsum.helpers.rand_equation
The python package [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/path_finding.html) has the function [opt_einsum.helpers.rand_equation](https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/helpers.py) which generates random TNs in einsum syntax. Those TNs are usually not connected (meaning they can be split into two or more independent TNs) and do not contain hyper edges, self edges, etc.
Our function random_tensor_network is an adaptation of this function and generates connected random TNs using the same input syntax.
Additionally, our function random_tensor_hyper_network generates connected random THNs which can be customized to fit all cases that can occur in an einsum expression (to the best of our knowledge).
