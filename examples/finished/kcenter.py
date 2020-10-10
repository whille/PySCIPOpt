"""
kcenter.py:  model for solving the k-center problem.

"minimization of the maximum value" problem, usally slow:
select k facility positions such that the maximum distance
of each vertex in the graph to a facility is minimum.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum
from kmedian import make_data, show_facility


def kcenter(I, J, c, k):
    """kcenter -- minimize the maximum travel cost from customers to k facilities.
    Parameters:
        - I: set of customers
        - J: set of potential facilities
        - c[i,j]: cost of servicing customer i from facility j
        - k: number of facilities to be used
    Returns a model, ready to be solved.
    """
    model = Model("k-center")
    z = model.addVar(vtype="C", name="z")
    x, y = {}, {}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    for i in I:
        model.addCons(quicksum(x[i, j] for j in J) == 1, "Assign(%s)" % i)
        for j in J:
            model.addCons(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))
            model.addCons(c[i, j] * x[i, j] <= z, "Max_x(%s,%s)" % (i, j))
    model.addCons(quicksum(y[j] for j in J) == k, "Facilities")
    model.setObjective(z, "minimize")
    model.data = x, y
    return model


def solve_kcenter(I, J, c, k, EPS=1.e-6):
    model = kcenter(I, J, c, k)
    # model.Params.Threads = 1
    model.optimize()
    x, y = model.data
    edges = [(i, j) for (i, j) in x if model.getVal(x[i, j]) > EPS]
    facilities = [j for j in y if model.getVal(y[j]) > EPS]
    return facilities, edges


if __name__ == "__main__":
    n = 100
    m = n
    I, J, c, x_pos, y_pos = make_data(n, m, same=True)
    k = 5
    facilities, edges = solve_kcenter(I, J, c, k)
    show_facility(I, J, c, x_pos, y_pos, facilities, edges)
