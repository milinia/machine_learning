import sys
import networkx as nx
import matplotlib.pyplot as plt
import random

def find_neighbor_edges(vertex, edges):
    neighbor_edges = []
    for edge in edges:
        if (edge[0] == vertex) | (edge[1] == vertex):
            neighbor_edges.append(edge)
    return neighbor_edges


def find_min_edge(edges):
    min_weight = sys.maxsize
    min_edge = ()
    for edge in edges:
        if edge[2]['weight'] < min_weight:
            min_weight = edge[2]['weight']
            min_edge = edge
    return min_edge

def find_max_edge(edges):
    max_weight = -sys.maxsize
    max_edge = ()
    for edge in edges:
        if edge[2]['weight'] > max_weight:
            max_weight = edge[2]['weight']
            max_edge = edge
    return max_edge


def delete_edges(edges, clusters_num):
    for i in range(clusters_num - 1):
        max_edge = find_max_edge(edges)
        edges.remove(max_edge)
    return edges

def find_shortest_unclosed_path(graph, clusters_num, n):
    new_edges = []
    taken_vertex = set()
    edges = list(graph.edges(data=True))
    print(edges)
    min_edge = find_min_edge(edges)
    new_edges.append(min_edge)
    taken_vertex.update([min_edge[0], min_edge[1]])
    edges.remove(min_edge)
    while edges:
        if len(taken_vertex) == n:
            break
        neighbor_edges = []
        for vertex in taken_vertex:
            neighbor_edges.extend(find_neighbor_edges(vertex, edges))
        min_edge = find_min_edge(neighbor_edges)
        if min_edge[0] in taken_vertex and min_edge[1] in taken_vertex:
            edges.remove(min_edge)
        else:
            new_edges.append(min_edge)
            taken_vertex.update([min_edge[0], min_edge[1]])
            edges.remove(min_edge)
    new_graph = create_graph_from_edges(n, new_edges)
    draw_graph(new_graph)
    new_edges = delete_edges(new_edges, clusters_num)
    new_graph = create_graph_from_edges(n, new_edges)
    return new_graph

def create_graph_from_array(n, array):
    new_graph = nx.Graph()
    nodes = range(n)
    new_graph.add_nodes_from(nodes)
    new_graph.add_weighted_edges_from(array)
    return new_graph

def create_graph_from_edges(n, edges):
    new_graph = nx.Graph()
    nodes = range(n)
    new_graph.add_nodes_from(nodes)
    new_graph.add_edges_from(edges)
    return new_graph

def draw_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=15, width=3)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()

if __name__ == '__main__':
    edges = []
    n = 5
    for i in range(n):
        for j in range(i + 1, n):
            if random.randint(0, 1) == 1:
                weight = 1
                edges.append((i, j, random.randint(1, 10)))

    G = create_graph_from_array(5, edges)
    print(G)
    draw_graph(G)
    new_graph = find_shortest_unclosed_path(G, 2, n)
    draw_graph(new_graph)

