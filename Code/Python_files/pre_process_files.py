""" Functions for preprocessing the graph files.

Files should follow certain contracts.
"""
import numpy as np

def check_for_double_edges():
    """ Check a graph file for double edges.

    Edge file should be name "out_graph.txt". The first row
    is the number of the total nodes in tha graph. Each of the
    next rows describes a directed edge from source node to
    target node. The form is <int> <int> or <int>\t<int>

    """
    # Edge list, set for no dublicates.
    edge_list = set()

    with open("out_graph.txt", "r") as file_one:
        # Read first row number of nodes. 
        file_one.readline()
        # For every line.
        for line in file_one:
            # Read edge.
            edge = (int(line.split()[0]), int(line.split()[1]))
            # Check if edge already exists in edge list.
            if edge in edge_list:
                print("-----------Double Edge Exists-----------")
                return
            # Add edge to edge_list.
            edge_list.add(edge)

def remove_double_edges():
    """ Remove double edges from an edge file.

    Edge file should be name "out_graph.txt". The first row
    is the number of the total nodes in tha graph. Each of the
    next rows describes a directed edge from source node to
    target node. The form is <int> <int> or <int>\t<int>

    """
    # Edge list, set for no dublicates.
    edge_list = set()
    n_nodes = 0

    with open("out_graph.txt", "r") as file_one:
        # Read number of nodes. 
        n_nodes = int(file_one.readline())
        # For every line.
        for line in file_one:
            # Read edge.
            edge = line.split()
            # Add edge to edge_list.
            edge_list.add((int(edge[0]), int(edge[1])))

    with open("out_graph.txt", "w") as file_one:
        file_one.write("%d\n" %n_nodes)
        for edge in edge_list:
            file_one.write("%d\t%d\n" %(edge[0], edge[1]))
     
def change_categories():
    """ Exchange Categories so as 1 to be the unfavoured.

    Edge file should be name "out_graph.txt". The first row
    is the number of the total nodes in tha graph. Each of the
    next rows describes a directed edge from source node to
    target node. The form is <int> <int> or <int>\t<int>

    """
    # Edge list, set for no dublicates.
    cat_list = set()
    n_com = 0

    with open("out_community.txt", "r") as file_one:
        # Read number of nodes. 
        n_com = int(file_one.readline())
        # For every line.
        for line in file_one:
            # Read edge.
            edge = line.split()
            # Add edge to edge_list.
            cat_list.add((int(edge[0]), 1 - int(edge[1])))

    with open("out_community.txt", "w") as file_one:
        file_one.write("%d\n" %n_com)
        for cat in cat_list:
            file_one.write("%d\t%d\n" %(cat[0], cat[1]))

def clean_dataset():
    """ Clean a dataset so as to follow the contracts for our algorithms.

    Parameters:
        edge file:
        community file:
    Output:
        edge file:
        community_file:
    """
    pass

check_for_double_edges()
remove_double_edges()