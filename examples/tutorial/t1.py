#!/usr/bin/python3
# https://imada.sdu.dk/~marco/DM545/Training/dm545_lab_scip.pdf

import pyscipopt as pso
from pyscipopt import Model
model = Model("prod", enablepricing=True)
# Create decision variables
x1 = model.addVar(name="x1", vtype="C", lb=0.0, ub=None, obj=5.0, pricedVar=False)
x2 = model.addVar("x2", "C", 0, None, 6)  # arguments by position
x3 = model.addVar(name="x3")  # arguments by deafult: lb=0.0, ub=None, obj=0.
# Unecessary if we had written the obj coefficient for all vars above
model.setObjective(5.0 * x1 + 6.0 * x2 + 8.0 * x3, "maximize")
# Add constraints to the model
model.addCons(6.0 * x1 + 5.0 * x2 + 10.0 * x3 <= 60.0, "c1")
model.addCons(8.0 * x1 + 4.0 * x2 + 4.0 * x3 <= 40.0, "c2")
model.addCons(4.0 * x1 + 5.0 * x2 + 6.0 * x3 <= 50.0, "c3")

# Solve
model.optimize()
# Let’s print the solution
if model.getStatus() == "optimal":
    print("Optimal value:", model.getObjVal())
    for v in model.getVars():
        print(v.name, " = ", model.getVal(v), ', reduced cost =', model.getVarRedcost(v))
else:
    print("Problem could not be solved to optimality")


def change_para():
    model.setPresolve(pso.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pso.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    # let’s use the primal simplex
    model.setCharParam("lp/initalgorithm", "p")
    model.setCharParam("lp/pricing", "f")


def output():
    # Write the set of SCIP parameters and their settings.
    model.writeParams("param.set")
    # Write the instantiated model to a file
    model.writeProblem("prod1_scip.lp")     # lp format


# Let’s print slack and dual variables
for c in model.getConss():
    print(c.name, model.getSlack(c), model.getDualsolLinear(c))
