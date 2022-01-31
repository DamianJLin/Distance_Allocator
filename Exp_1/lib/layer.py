def layer(subcircuit):
    """
    Generates the gates (edges) in the top layer of subcircuit (all gates that can be run parallel to subcircuit[0]).

    :param subcircuit: List[List[Int]] the sublist of gates to generate from.
    :return: None
    """
    u_init, v_init = subcircuit[0]
    yield subcircuit[0]
    k = 1
    while u_init not in subcircuit[k] and v_init not in subcircuit[k]:
        yield subcircuit[k]
        k += 1

    return
