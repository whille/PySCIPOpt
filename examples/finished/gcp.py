# @file gcp.py
# @brief model for the graph coloring problem
"""
Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

import time
import sys
from pyscipopt import Model, quicksum
from graph_utils import make_data, show_solution


def gcp(V, E, K):
    """gcp -- model for minimizing the number of colors in a graph
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
        - K: upper bound on the number of colors
    Returns a model, ready to be solved.
    """
    model = Model("gcp")
    x, y = {}, {}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)" % k)
        for i in V:
            x[i, k] = model.addVar(vtype="B", name="x(%s,%s)" % (i, k))

    for i in V:
        model.addCons(
            quicksum(x[i, k] for k in range(K)) == 1, "AssignColor(%s)" % i)

    for (i, j) in E:
        for k in range(K):
            model.addCons(x[i, k] + x[j, k] <= y[k],
                          "NotSameColor(%s,%s,%s)" % (i, j, k))

    model.setObjective(quicksum(y[k] for k in range(K)), "minimize")
    model.data = x, y
    return model


def gcp_low(V, E, K):
    model = gcp(V, E, K)
    y = model.data[1]
    for k in range(K - 1):
        model.addCons(y[k] >= y[k + 1], "LowColor(%s)" % k)
    return model


def gcp_sos(V, E, K):
    """gcp_sos -- model for minimizing the number of colors in a graph
    (use sos type 1 constraints)
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
        - K: upper bound to the number of colors
    Returns a model, ready to be solved.
    """
    model = Model("gcp - sos constraints")
    x, y = {}, {}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)" % k)
        for i in V:
            x[i, k] = model.addVar(vtype="B", name="x(%s,%s)" % (i, k))
    for i in V:
        model.addCons(quicksum(x[i, k] for k in range(K)) == 1, "AssignColor(%s)" % i)
        model.addConsSOS1([x[i, k] for k in range(K)])
    for (i, j) in E:
        for k in range(K):
            model.addCons(x[i, k] + x[j, k] <= y[k], "NotSameColor(%s,%s,%s)" % (i, j, k))
    for k in range(K - 1):
        model.addCons(y[k] >= y[k + 1], "LowColor(%s)" % k)
    model.setObjective(quicksum(y[k] for k in range(K)), "minimize")
    model.data = x, y
    return model


if __name__ == "__main__":
    V, E = make_data(20, .5)
    K = 10  # upper bound to the number of colors
    print("n,K=", len(V), K)

    model = gcp_low(V, E, K)
    model.optimize()
    print("Optimal value:", model.getObjVal())
    x, y = model.data
    show_solution(K, V, x, model)
    models = [gcp, gcp_low, gcp_sos]
    cpu = {}
    N = 10  # number of observations
    print('calc time compares(less is better):')
    print("#size\t%s\t%s\t%s" % tuple(m.__name__ for m in models))
    for size in range(10, 30):
        stats = []
        K = size
        for prob in [0.1]:
            for m in models:
                name = m.__name__
                if not (name, size - 1, prob) in cpu or cpu[
                        name, size - 1,
                        prob] < 100:  # cpu.has_key((name,size-1,prob))
                    cpu[name, size, prob] = 0.
                    for t in range(N):
                        tinit = time.time()
                        V, E = make_data(size, prob, t)
                        model = m(V, E, K)
                        model.hideOutput()  # silent mode
                        model.optimize()
                        assert model.getObjVal() >= 0 and model.getObjVal() <= K
                        tend = time.time()
                        cpu[name, size, prob] += tend - tinit
                    cpu[name, size, prob] /= N
                else:
                    cpu[name, size, prob] = "-"
                stats.append(f"{cpu[name, size, prob]: .3f}")
        s_stat = '\t'.join(stats)
        print(f"{size}\t{s_stat}")
        sys.stdout.flush()
