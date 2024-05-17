from typing import Union, Optional, Tuple, Dict, Collection
from numpy.random import default_rng
from opt_einsum.paths import _find_disconnected_subgraphs #NOTE: only for testing

PathType = Collection[Tuple[int, ...]]

def get_symbol(i: int) -> str:
    """
    Get a symbol (str) corresponding to int i - runs through the usual 52
    letters before using other unicode characters.
    """
    alphabetic_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    if i< 52:
        return alphabetic_symbols[i]
    return chr(i+140)

def random_tensor_network(
    number_of_tensors: int,
    regularity: float,
    number_of_output_indices: int = 0,
    min_axis_size: int = 2,
    max_axis_size: int = 10,
    seed: Optional[int] = None,
    global_dim: bool = False,
    return_size_dict: bool = False,
    ) -> Union[Tuple[str, PathType, Dict[str, int]], Tuple[str, PathType]]:
    """Generate a random connected Tensor Network (TN). Returns an einsum expressions string representing the TN, shapes of the tensors and optionally a dictionary containing the index sizes.

    Parameters
    ----------
    number_of_tensors : int
        Number of tensors/arrays in the TN.
    regularity : float
        'Regularity' of the TN. This determines how
        many indices/axes each tensor shares with others on average (not counting output indices).
    number_of_output_indices : int, optional
        Number of output indices/axes (i.e. the number of non-contracted indices).
        Defaults to 0, i.e., a contraction resulting in a scalar.
    min_axis_size : int, optional
        Minimum size of an axis/index (dimension) of the tensors.
    max_axis_size : int, optional
        Maximum size of an axis/index (dimension) of the tensors.
    seed: int, optional
        If not None, seed numpy's random generator with this.
    global_dim : bool, optional
        Add a global, 'broadcast', dimension to every operand.
    return_size_dict : bool, optional
        Return the mapping of indices to sizes.

    Returns
    -------
    eq : str
        The einsum expression string.
    shapes : list[tuple[int]]
        The shapes of the tensors/arrays.
    size_dict : dict[str, int]
        The dict of index sizes, only returned if ``return_size_dict=True``.

    Examples #TODO
    --------
    >>> eq, shapes = rand_equation(n=10, reg=4, number_of_output_indices=5, seed=42)
    >>> eq
    'oyeqn,tmaq,skpo,vg,hxui,n,fwxmr,hitplcj,kudlgfv,rywjsb->cebda'

    >>> shapes
    [(9, 5, 4, 5, 4),
     (4, 4, 8, 5),
     (9, 4, 6, 9),
     (6, 6),
     (6, 9, 7, 8),
     (4,),
     (9, 3, 9, 4, 9),
     (6, 8, 4, 6, 8, 6, 3),
     (4, 7, 8, 8, 6, 9, 6),
     (9, 5, 3, 3, 9, 5)]
    """

    # create rng
    if seed is None:
        rng = default_rng()
    else: 
        rng = default_rng(seed)

    # total number of indices
    number_of_indices = int(number_of_tensors * regularity) // 2 + number_of_output_indices #NOTE: output indices are not counted for degree.
    tensors = []
    output = []

    size_dict = {get_symbol(i): rng.integers(min_axis_size, max_axis_size + 1) for i in range(number_of_indices)}

    # generate TN as einsum string
    for index_number, index in enumerate(size_dict):
        # generate first two tensors connected by an edge to start with
        if index_number == 0:
            tensors.append(index)
            tensors.append(index)
            continue

        # generate a bound/edge
        if index_number < number_of_indices - number_of_output_indices:

            # add tensors and connect to existing tensors, until number of tensors is reached
            if len(tensors) < number_of_tensors:
                connect_to_tensor = rng.integers(0, len(tensors))
                tensors[connect_to_tensor] += index
                tensors.append(index)
            # add edges between existing tensors    
            else:
                tensor_1 = rng.integers(0, len(tensors))
                tensor_2 = rng.integers(0, len(tensors))
                while tensor_2 == tensor_1:
                    tensor_2 = rng.integers(0, len(tensors))
                tensors[tensor_1] += index
                tensors[tensor_2] += index

        # generate an output index
        else:
            tensor = rng.integers(0, len(tensors))
            tensors[tensor] += index
            output += index

    # check specs
    assert len(tensors) == number_of_tensors, f"number generated tensors/tensors = {len(tensors)} does not match number_of_tensors = {number_of_tensors}."
    assert len(output) == number_of_output_indices, f"number of generated output indices = {len(output)} does not match number_of_output_indices = {number_of_output_indices}."
    # assert len(_find_disconnected_subgraphs([set(input) for input in tensors], set(output))) == 1, "the generated graph is not connected." # check if graph is connected

    # possibly add the same global dim to every arg
    if global_dim:
        gdim = get_symbol(number_of_indices)
        size_dict[gdim] = rng.integers(min_axis_size, max_axis_size + 1)
        for i in range(number_of_tensors):
            tensors[i] += gdim
        output += gdim

    # randomly transpose the output indices and form equation
    output = "".join(rng.permutation(output))
    tensors = ["".join(rng.permutation(list(tensor))) for tensor in tensors]
    eq = "{}->{}".format(",".join(tensors), output)

    # make the shapes
    shapes = [tuple(size_dict[ix] for ix in op) for op in tensors]

    ret = (eq, shapes)

    if return_size_dict:
        ret += (size_dict, )

    return ret

def random_tensor_hyper_network( #LATER
    
    number_of_tensors: int,
    regularity: float,
    max_tensor_order: int = None,
    max_edge_order: int = 2,
    diagonals_in_hyper_edges: bool = True, #TODO write test for this
    number_of_output_indices: int = 0,
    max_output_index_order: int = 1,
    diagonals_in_output_indices: bool = True, #TODO write test for this
    number_of_self_edges: int = 0,
    max_self_edge_order: int = 2,
    number_of_single_summation_indices: int = 0,
    #output_index_as_diagonal_of_tensor: bool = True, #LATER maybe
    min_axis_size: int = 2,
    max_axis_size: int = 10,
    seed: Optional[int] = None,
    #global_dim: bool = False, #TODO
    return_size_dict: bool = False,
    ) -> Union[Tuple[str, PathType, Dict[str, int]], Tuple[str, PathType]]:

    """Generate a random contraction and shapes.

    Parameters
    ----------
    number_of_tensors : int
        Number of tensors/arrays in the TN.
    regularity : float
        'Regularity' of the TN. This determines how
        many indices/axes each tensor shares with others on average (not counting output indices, self edges and single summation indices).
    max_tensor_order: int = None, optional
        The maximum order (number of axes/dimensions) of the tensors. If ``None``, use an upper bound calculated from other parameters.
    max_edge_order: int, optional
        The maximum order of hyperedges.
    diagonals_in_hyper_edges: bool = True,
        Whether diagonals can appear in hyper edges, e.g. in "aab,ac,ad -> bcd" a is a hyper edge with a diagonal in the first tensor.
    number_of_output_indices : int, optional
        Number of output indices/axes (i.e. the number of non-contracted indices).
        Defaults to 0, i.e., a contraction resulting in a scalar.
    max_output_index_order: int = 1, optional
        Restricts the number of times the same output index can occur.
    diagonals_in_output_indices: bool = True,
        Whether diagonals can appear in output indices, e.g. in "aab,ac -> abc" a is an output index with a diagonal in the first tensor.
    number_of_self_edges: int = 0, optional
        The number of self edges/traces (e.g. in "ab,bcdd->ac" d represents a self edge).
    max_self_edge_order: int = 2, optional
        The maximum order of a self edge e.g. in "ab,bcddd->ac" the self edge represented by d has order 3.
    number_of_single_summation_indices: int = 0, optional
        The number of indices that are not connected to any other tensors and do not show up in the ouput (e.g. in "ab,bc->c" a is a single summation index).
    min_axis_size : int, optional
        Minimum size of an axis/index (dimension) of the tensors.
    max_axis_size : int, optional
        Maximum size of an axis/index (dimension) of the tensors.
    seed: int, optional
        If not None, seed numpy's random generator with this.
    global_dim : bool, optional
        Add a global, 'broadcast', dimension to every operand.
    return_size_dict : bool, optional
        Return the mapping of indices to sizes.

    Returns
    -------
    eq : str
        The einsum expression string.
    shapes : list[tuple[int]]
        The shapes of the tensors/arrays.
    size_dict : dict[str, int]
        The dict of index sizes, only returned if ``return_size_dict=True``.

    Example #TODO
    --------
    >>> eq, shapes = rand_equation_hyper(number_of_tensors=10, regularity=3, max_tensor_order=8, max_edge_order=4, number_of_output_indices=5, max_output_index_order=3, min_axis_size=2, max_axis_size=5, connected = True)
    >>> eq
    'a,jha,ia,ndeb,dlic,d,mbc,iim,bmhlkfj,hg->bcdea'

    >>> shapes
    [(4,), (5, 3, 4), (5, 4), (5, 3, 3, 3), (3, 3, 5, 4), (3,), (2, 3, 4), (5, 5, 2), (3, 2, 3, 3, 4, 4, 5), (3, 5)]
    """

    # handle 'None' in tensors
    if max_tensor_order == None:
        max_tensor_order = int((number_of_tensors - 1) * regularity + number_of_self_edges * max_self_edge_order + number_of_output_indices * max_output_index_order + number_of_single_summation_indices) # in the worst case, everything gets attached to one tensor
    # create rng
    if seed is None:
        rng = default_rng()
    else: 
        rng = default_rng(seed)

    # check if tensors make sense
    assert regularity <= max_tensor_order, 'regularity is higher than chosen max_tensor_order' 

    # check if max_tensor_order suffices to fit all connecting edge, output indices, self edges and single summation indices
    assert max_tensor_order * number_of_tensors >= int(regularity * number_of_tensors) + number_of_output_indices + number_of_self_edges * 2 + number_of_single_summation_indices, f"the max_tensor_order*number_of_tensors =  {max_tensor_order * number_of_tensors} is not high enough to fit all {int(regularity*number_of_tensors)} connecting indices, {number_of_output_indices} output_indices, {2 * number_of_self_edges} indices of self_edges and {number_of_single_summation_indices} single summation indices." 
    
    number_of_connecting_indices = int(number_of_tensors * regularity) # how many indices make up the underlying hypergraph. To this hyperedges contribute += order, These do not contribute: self edges, summation/single contr. and out edges
    number_of_spaces = number_of_tensors * max_tensor_order # spaces = total number of indices that can be placed in tensors such that the max order is satisfied
    number_of_reserved_spaces = 2 * number_of_tensors + 2 * number_of_self_edges + number_of_single_summation_indices + number_of_output_indices # how many spaces are at least neccessary to fulfil the given specifications
    
    number_of_connecting_indices_to_do = number_of_connecting_indices # keep track of how may connections are left to do
    tensors = []
    output = ""
    not_max_order_tensors = [] # keeps track of existing tensors to which indices can be added to
    free_spaces_in_not_max_order_tensors = 0 # tracks how many spaces are free in not_max_order_tensors

    # ADD ALL TENSORS such that they are connected to the graph with (hyper-)edges

    # create start tensors
    tensors.append(get_symbol(0))
    tensors.append(get_symbol(0))
    not_max_order_tensors.append(0)
    not_max_order_tensors.append(1)
    number_of_reserved_spaces -= 4 # took care of two tensors
    free_spaces_in_not_max_order_tensors += 2 * (max_tensor_order - 1) # one index in both tensors

    for tensor_number in range(2, number_of_tensors):
        index = get_symbol(tensor_number - 1)

        # determine order of hyperedge
        number_of_tensors_to_do = number_of_tensors - tensor_number
        non_reserved_spaces = number_of_spaces - number_of_reserved_spaces

        # determine max order
        if diagonals_in_hyper_edges:
            max_order = min(free_spaces_in_not_max_order_tensors, max_edge_order, number_of_connecting_indices_to_do, non_reserved_spaces) # can only connect as many times to not_max_order_tensors, as there are free spaces, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved
        else:
            max_order = min(len(not_max_order_tensors), max_edge_order, number_of_connecting_indices_to_do - 2 * number_of_tensors_to_do, non_reserved_spaces) # we can only connect to existing tensors which do not have the max order, respect the max edge order, need to make sure we have enough indices left for the other tensors and need to respect the number of not reserved spaces

        order = rng.integers(2, max_order + 1)

        # determine tensors to connect to
        if diagonals_in_hyper_edges:
            for index_number in range(order):
                # fist connect to already existing tensors
                if index_number == 1:
                    tensors.append(index)
                    not_max_order_tensors.append(len(tensors) - 1)
                    continue
                tensor = rng.choice(not_max_order_tensors) #NOTE: we can get diagonals over one tensor in a hyperedge
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        else:
            # connect to other tensors
            connect_to_tensors = rng.choice(not_max_order_tensors, size = order - 1, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
            # connect to new tensor
            tensors.append(index)
            not_max_order_tensors.append(len(tensors) - 1)
        
        # update tracking
        number_of_connecting_indices_to_do -= order
        number_of_reserved_spaces -= 2 # took care of one tensor
        free_spaces_in_not_max_order_tensors += max_tensor_order - order # added one new tensor but filled order spaces

    assert len(tensors) == number_of_tensors, f"The number of created tensors/tensors = {len(tensors)} does not match number_of_tensors = {number_of_tensors}."

    # REMAINING CONNECTIONS between tensors
    number_of_used_indices = number_of_tensors - 1
    non_reserved_spaces = number_of_spaces - number_of_reserved_spaces

    while number_of_connecting_indices_to_do > 0:
        index = get_symbol(number_of_used_indices)

        # determine order of hyperedge:
        if diagonals_in_hyper_edges:
            max_order = min(free_spaces_in_not_max_order_tensors, max_edge_order, number_of_connecting_indices_to_do, non_reserved_spaces) # can only fill free spaces in tensors that do not have max order, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved
        else:
            max_order = min(len(not_max_order_tensors), max_edge_order, number_of_connecting_indices_to_do, non_reserved_spaces) # can only connect to tensors that do not have max order, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved

        order = rng.integers(2, max_order + 1)
        # make sure that number_of_connecting indices to do is not left at 1
        while number_of_connecting_indices_to_do - order == 1:
            order = rng.integers(2, max_order + 1)

        # determine tensors to connect to
        if order == 2: # no pure self edges here
            connect_to_tensors = rng.choice(not_max_order_tensors, size = 2, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        elif diagonals_in_hyper_edges:
            for _ in range(order):
                tensor = rng.choice(not_max_order_tensors) #NOTE: we can get diagonals over one tensor in a hyperedge
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        else:
            # connect to tensors
            connect_to_tensors = rng.choice(not_max_order_tensors, size = order, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)

        # update tracking
        number_of_connecting_indices_to_do -= order
        number_of_used_indices += 1
        free_spaces_in_not_max_order_tensors -= order # filled order spaces

    # check if all connections have been made
    assert number_of_connecting_indices_to_do == 0, f"The number of created connections = {number_of_connecting_indices-number_of_connecting_indices_to_do} does not fit regularity * number_of_tensors = {regularity * number_of_tensors}."

    # SELF EDGES
    for _ in range(number_of_self_edges):
        index = get_symbol(number_of_used_indices)

        # determine order of self edge:
        non_reserved_spaces = number_of_spaces - number_of_reserved_spaces
        max_order = min(len(not_max_order_tensors), max_self_edge_order, non_reserved_spaces) # respect max order of tensors, max order of output index,  number of output indices left to do and not reserved spaces
        order = rng.integers(2, max_order + 1)

        # determine tensor for self edge
        tensor = rng.choice(not_max_order_tensors)

        # make sure the tensor has enough spaces left
        while len(tensors[tensor]) + order > max_tensor_order:
            order = rng.integers(2, max_order + 1)
            tensor = rng.choice(not_max_order_tensors)
        
        tensors[tensor] += index * order
            
        # check if max order is reached
        if len(tensors[tensor]) == max_tensor_order:
            not_max_order_tensors.remove(tensor)

        # update tracking
        number_of_reserved_spaces -= 2 # took care of one self edge
        number_of_used_indices += 1
        free_spaces_in_not_max_order_tensors -= order # filled order spaces


    # SINGLE SUMMATION INDICES
    for _ in range(number_of_single_summation_indices):
        index = get_symbol(number_of_used_indices)

        tensor = rng.choice(not_max_order_tensors)
        tensors[tensor] += index

        # check if max order is reached
        if len(tensors[tensor]) == max_tensor_order:
            not_max_order_tensors.remove(tensor)

        # update tracking #TODO
        number_of_reserved_spaces -= 1 # took care of one single summation index
        number_of_used_indices += 1
        free_spaces_in_not_max_order_tensors -= 1 # filled 1 space

        
    # OUTPUT INDICES
    for output_index_number in range(1, number_of_output_indices + 1):
        index = get_symbol(number_of_used_indices)

        # determine order of output index:
        number_of_output_indices_to_do = number_of_output_indices - output_index_number
        non_reserved_spaces = number_of_spaces - number_of_reserved_spaces
        
        if diagonals_in_output_indices:
            max_order = min(max_output_index_order, free_spaces_in_not_max_order_tensors, non_reserved_spaces) # can only fill free spaces in tensors that do not have max order, need to respect the max edge order, how many connections we can still do and how many spaces are not reserved
        else:
            max_order = min(max_output_index_order, len(not_max_order_tensors), non_reserved_spaces) # respect max order of output index, number of free spaces in tensors and number of output indices left to do (non_reserved_spaces)

        order = rng.integers(1, max_order + 1)

        # determine tensors to connect to
        output += index
        if diagonals_in_output_indices:
            for _ in range(order):
                tensor = rng.choice(not_max_order_tensors) #NOTE:we can get diagonals over one tensor in an output index
                tensors[tensor] += index
                
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)
        else:
            connect_to_tensors = rng.choice(not_max_order_tensors, size = order, replace = False, shuffle = False)
            for tensor in connect_to_tensors:
                tensors[tensor] += index
                # check if max order is reached
                if len(tensors[tensor]) == max_tensor_order:
                    not_max_order_tensors.remove(tensor)

        # update tracking
        number_of_used_indices += 1
        number_of_reserved_spaces -= 1 # took care of one output index
        free_spaces_in_not_max_order_tensors -= order # filled order spaces

    #TODO: global dim

    # # possibly add the same global dim to every arg
    # if global_dim:
    #     gdim = get_symbol(number_of_indices)
    #     size_dict[gdim] = rng.integers(min_axis_size, max_axis_size + 1)
    #     for i in range(number_of_tensors):
    #         tensors[i] += gdim
    #     output += gdim

    # check length of output and that all specifications are fulfilled
    assert number_of_reserved_spaces == 0, f"{number_of_reserved_spaces} spaces are still reserved."
    assert len(output) == number_of_output_indices

    # randomly shuffle outputs and tensors
    output = "".join(rng.permutation(list(output)))

    # Test if hypergraph is connected #NOTE connected in opt einsum sense means shared output indices is not a connection. In the sense of cotengra's Hypergraph this would be a connection.
    assert len(_find_disconnected_subgraphs([set(input) for input in tensors], set(output))) == 1, f"the generated hypergraph has {len(_find_disconnected_subgraphs([set(input) for input in tensors], set(output)))} components." #TODO comment out later

    tensors = ["".join(rng.permutation(list(input))) for input in tensors]
    # form equation
    eq = "{}->{}".format(",".join(tensors), output)

    # get random size for an index
    size_dict = {get_symbol(index): rng.integers(min_axis_size, max_axis_size + 1) for index in range(number_of_used_indices)}

    # make the shapes
    shapes = [tuple(size_dict[idx] for idx in op) for op in tensors]

    ret = (eq, shapes)

    if return_size_dict:
        return ret + (size_dict,)
    else:
        return ret