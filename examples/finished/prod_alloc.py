#!/usr/bin/python3
# https://imada.sdu.dk/~marco/DM545/Training/dm545_lab_scip.pdf

from pyscipopt import Model
from util import show_dual


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

# let’s use the primal simplex
model.setCharParam("lp/initalgorithm", "p")
model.setCharParam("lp/pricing", "f")
show_dual(model)
