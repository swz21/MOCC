from math import ceil
import sys
from itertools import product

class Vertex:
    def __init__(self, weights):
        self.weights = weights
        self.visited = False
        self.edges = []
        self.d = []

    def add_edge(self, vertex):
        self.edges.append(vertex)

def generate_vertices():
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    vertices = []
    
    for w1, w2, w3 in product(weights, repeat=3):
        if abs(w1 + w2 + w3 - 1.0) < 1e-9:
            vertices.append(Vertex([w1, w2, w3]))
    
    return vertices

def generate_edges(vertices):
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            if i != j and is_neighbor(v1.weights, v2.weights):
                v1.add_edge(v2)

def is_neighbor(w1, w2):
    count = 0
    for i in range(3):
        if abs(w1[i] - w2[i]) < 1e-9:
            count += 1
        elif abs(w1[i] - w2[i] - 0.1) < 1e-9 or abs(w1[i] - w2[i] + 0.1) < 1e-9:
            continue
        else:
            return False
    return count == 1

def neighborhood_based_objective_sorting_algorithm(G, O):
    V = G['vertices']
    
    L = []
    
    for v in V:
        v.visited = False
        v.d = [sys.maxsize] * len(O)
    
    for i in range(len(O)):
        for v in V:
            if v in O[i].edges:
                v.d[i] = 1
            else:
                v.d[i] = sys.maxsize
    
    for i in range(len(O)):
        visits = ceil(len(V) / len(O))
        if not O[i].visited:
            L.append(O[i])
            O[i].visited = True
            visits -= 1
        
        while visits > 0 and len(L) < len(V):
            u = min((v for v in V if not v.visited), key=lambda v: v.d[i])
            u.visited = False
            L.append(u)
            u.visited = True
            visits -= 1
            
            for w in u.edges:
                if not w.visited and u.d[i] + 1 < w.d[i]:
                    w.d[i] = u.d[i] + 1

    return L

vertices = generate_vertices()
generate_edges(vertices)

G = {
    'vertices': vertices,
}

bootstrapped_vertices = [v for v in vertices if v.weights in [[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]]]

sorted_list = neighborhood_based_objective_sorting_algorithm(G, bootstrapped_vertices)
print([v.weights for v in sorted_list])