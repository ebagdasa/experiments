import sklearn
import numpy as np
import os


def get_edges(name):
    edges = dict()
    with open(name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            edges[tuple(line.split(' '))] = 1
    return edges

def get_embeddings(name):
    embeddings = list()
    with

