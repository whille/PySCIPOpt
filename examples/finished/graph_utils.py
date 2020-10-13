#!/usr/bin/env python

import random


def make_data(n, prob, seed=1):
    """make_data: prepare data for a random graph
    Parameters:
       - n: number of vertices
       - prob: probability of existence of an edge, for each pair of vertices
    Returns a tuple with a list of vertices and a list edges.
    """
    random.seed(seed)
    V = range(1, n + 1)
    E = [(i, j) for i in V for j in V if i < j and random.random() < prob]
    return V, E


def show_solution(K, V, x, model):
    color = {}
    dic_k = {}
    for i in V:
        for k in range(K):
            if model.getVal(x[i, k]) > 0.5:
                color[i] = k
                dic_k.setdefault(k, [])
                dic_k[k].append(i)
    print("colors:", color)
    print(f'clusters: {dic_k}')
