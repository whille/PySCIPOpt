"""
kcenter_binary_search.py:  use bisection for solving the k-center problem

bisects the interval [0, max facility-customer distance] until finding a
distance such that all customers are covered, but decreasing that distance
by a small amount delta would leave some uncovered.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum
from kmedian import make_data, show_facility


def kcover(I, J, c, k):
    """kcover -- minimize the number of uncovered customers from k facilities.
    Parameters:
        - I: set of customers
        - J: set of potential facilities
        - c[i,j]: cost of servicing customer i from facility j
        - k: number of facilities to be used
    Returns a model, ready to be solved.
    """
    model = Model("k-center")
    z, y, x = {}, {}, {}
    for i in I:
        z[i] = model.addVar(vtype="B", name="z(%s)" % i, obj=1)
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    for i in I:
        model.addCons(quicksum(x[i, j] for j in J) + z[i] == 1, "Assign(%s)" % i)
        for j in J:
            model.addCons(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))
    model.addCons(quicksum(y[j] for j in J) == k, "Facilities")
    model.data = x, y, z
    return model


def solve_kcenter(I, J, c, k, delta):
    """solve_kcenter -- locate k facilities minimizing distance of most distant customer.
    Parameters:
        I - set of customers
        J - set of potential facilities
        c[i,j] - cost of servicing customer i from facility j
        k - number of facilities to be used
        delta - tolerance for terminating bisection
    Returns:
        - list of facilities to be used
        - edges linking them to customers
    """
    model = kcover(I, J, c, k)
    x, y, z = model.data

    facilities, edges = [], []
    LB = 0
    UB = max(c[i, j] for (i, j) in c)
    model.setObjlimit(0.1)
    model.hideOutput()
    visited = False
    while UB - LB > delta:
        theta = (UB + LB) / 2.
        print("\n\ncurrent theta:", theta)
        if visited:
            model.freeTransform()
        else:
            visited = True
        for j in J:
            for i in I:
                if c[i, j] > theta:
                    model.chgVarUb(x[i, j], 0.0)
                else:
                    model.chgVarUb(x[i, j], 1.0)

        # model.Params.OutputFlag = 0 # silent mode
        model.setObjlimit(.1)
        model.optimize()
        if model.getStatus() == "optimal":
            UB = theta
            facilities = [j for j in y if model.getVal(y[j]) > .5]
            edges = [(i, j) for (i, j) in x if model.getVal(x[i, j]) > .5]
        else:  # infeasibility > 0:
            LB = theta

    return facilities, edges


if __name__ == "__main__":
    n = 100
    m = n
    I, J, c, x_pos, y_pos = make_data(n, m, same=True)
    k = 5
    facilities, edges = solve_kcenter(I, J, c, k, delta=1.e-4)
    show_facility(I, J, c, x_pos, y_pos, facilities, edges)
