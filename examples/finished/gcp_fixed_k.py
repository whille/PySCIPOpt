# @file gcp_fixed_k.py
# @brief solve the graph coloring problem with fixed-k model
"""
Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from pyscipopt import Model, quicksum
from graph_utils import make_data, show_solution


def gcp_fixed_k(V, E, K):
    """gcp_fixed_k -- model for minimizing number of bad edges in coloring a graph
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
        - K: number of colors to be used
    Returns a model, ready to be solved.
    """
    model = Model("gcp - fixed k")

    x, z = {}, {}
    for i in V:
        for k in range(K):      # vetex i in color k or not
            x[i, k] = model.addVar(vtype="B", name="x(%s,%s)" % (i, k))
    for (i, j) in E:        # bad edge(same color) or not
        z[i, j] = model.addVar(vtype="B", name="z(%s,%s)" % (i, j))
    for i in V:
        model.addCons(quicksum(x[i, k] for k in range(K)) == 1, "AssignColor(%s)" % i)
        model.addConsSOS1([x[i, k] for k in range(K)])
    for (i, j) in E:
        for k in range(K):
            model.addCons(x[i, k] + x[j, k] <= 1 + z[i, j], "BadEdge(%s,%s,%s)" % (i, j, k))
    model.setObjective(quicksum(z[i, j] for (i, j) in E), "minimize")
    model.data = x
    return model


def solve_gcp(V, E):
    """solve_gcp -- solve the graph coloring problem with bisection and fixed-k model
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns tuple with number of colors used, and dictionary mapping colors to vertices
    """
    LB = 0
    UB = len(V)
    best_model = None
    print('binary search:')
    while UB - LB > 1:
        K = int((UB + LB) / 2)
        gcp = gcp_fixed_k(V, E, K)
        # gcp.setParam('OutputFlag', 0)   # silent mode
        # gcp.setParam('lp/cutoff', .1)
        gcp.setObjlimit(0.1)
        gcp.hideOutput()
        gcp.optimize()
        status = gcp.getStatus()
        print(f"\tLB={LB}, UB={UB}, K={K}, status:{status}")
        if status == "optimal":
            best_model = gcp
            UB = K
        else:
            LB = K
    return UB, best_model


if __name__ == "__main__":
    V, E = make_data(75, .25)
    K, model = solve_gcp(V, E)
    print("minimum number of colors:", K)
    show_solution(K, V, model.data, model)
