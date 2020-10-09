#!/usr/bin/env python
from pyscipopt import Model, quicksum

model = Model(enablepricing=True)
x = [model.addVar(vtype="B") for i in range(3)]
y = [model.addVar(vtype="B") for i in range(3)]
u = [model.addVar(vtype='B') for j in range(2)]
model.addConsOr(x, u[0])
model.addConsOr(y, u[1])
# z = [model.addVar(vtype="C", pricedVar=True) for i in range(3)]

# Set objective function
model.setObjective(-quicksum(x) - quicksum(y) + 2 * quicksum(u), "maximize")
model.optimize()

print(model.getStatus())
for i in range(3):
    print(x[i], model.getVal(x[i]), model.getVal(x[i]))
