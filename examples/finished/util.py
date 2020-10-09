#!/usr/bin/env python
import pyscipopt as pso


def prepare_dual(model):
    model.setPresolve(pso.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pso.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()


def show_sol(model):
    if model.getStatus() == "optimal":
        print("Optimal value:", model.getObjVal())
        for v in model.getVars():
            val, redcost = model.getVal(v), model.getVarRedcost(v)
            print(f"{v.name} = {val}, reduced cost = {redcost: .3f}")
    else:
        print("Problem could not be solved to optimality")


def output_model(model):
    # Write the set of SCIP parameters and their settings.
    model.writeParams("param.set")
    # Write the instantiated model to a file
    model.writeProblem("prod1_scip.lp")     # lp format


def show_slack(model):
    print('constrains info:')
    for c in model.getConss():
        slack, dual = model.getSlack(c), model.getDualsolLinear(c)
        print(f"{c.name}: slack = {slack:.3f}, dual = {dual:.3f}")


def show_dual(model):
    prepare_dual(model)
    model.optimize()
    show_sol(model)
    show_slack(model)
