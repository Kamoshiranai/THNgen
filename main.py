from generators.TN_generator import random_tensor_network
from generators.THN_generator import random_tensor_hyper_network

# Tensor Network Expressions

number_of_tensors = 10
regularity = 3.5
number_of_output_indices = 5
min_axis_size = 2
max_axis_size = 4

einsum_string, shapes, size_dict = random_tensor_network(
    number_of_tensors, 
    regularity, 
    number_of_output_indices, 
    min_axis_size, 
    max_axis_size, 
    return_size_dict = True,
    global_dim = False,
    seed = 12345
)

print("Generated Tensor Network:")
print("einsum string: " + einsum_string)
print("shapes: ", shapes)
print("size_dict: ", size_dict)
print("")

# Tensor Hyper Network Expressions

number_of_tensors = 10
regularity = 3.0
max_tensor_order = 10
max_edge_order = 3
diagonals_in_hyper_edges = True
number_of_output_indices = 5
max_output_index_order = 3
diagonals_in_output_indices = True
number_of_self_edges = 4
max_self_edge_order = 3
number_of_single_summation_indices = 3
global_dim = True
min_axis_size = 2
max_axis_size = 4

einsum_string, shapes, size_dict = random_tensor_hyper_network(
    number_of_tensors, 
    regularity, 
    max_tensor_order, 
    max_edge_order,
    diagonals_in_hyper_edges,
    number_of_output_indices, 
    max_output_index_order,
    diagonals_in_output_indices, 
    number_of_self_edges, 
    max_self_edge_order, 
    number_of_single_summation_indices, 
    global_dim,
    min_axis_size, 
    max_axis_size, 
    return_size_dict=True,
    seed = 12345
)

print("Generated Tensor Hyper Network:")
print("einsum string: " + einsum_string)
print("shapes: ", shapes)
print("size_dict: ", size_dict)

#TODO test with some random parameters