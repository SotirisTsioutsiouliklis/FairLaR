import numpy as np 
import sys
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.use('Agg')

no_of_bins = 10

#------New------
def get_cat_indexes():
    with open("out_graph.txt", "r") as file_one:
        n = int(file_one.readline())
        
    index = [False for i in range(n)]
    with open("out_community.txt", "r") as file_one:
        file_one.readline()
        for line in file_one:
            info = line.split()
            if int(info[1]) == 1:
                index[int(info[0])] = True

    return index

class Node():
    def __init__(self):
        self.id = 0
        self.out_neighbors = set()
        self.in_neighbors = set()
        self.x_to_give = 0.0
        self.excess_to_red = 0.0
        self.excess_to_blue = 0.0
        self.out_red = 0
        self.in_red = 0
        self.out_blue = 0
        self.in_blue = 0
        self.out_red_ratio = 0.0
        self.in_red_ratio = 0.0
        self.out_blue_ratio = 0.0
        self.in_blue_ratio = 0.0
        self.excess_to_red_individual = 0.0 # Neighborhood.
        self.excess_to_blue_individual = 0.0 # Neighborhood.
        self.importance_in_community = 0.0

    def set_node_id(number):
        self.id = number

    def set_out_red_ratio(self, ratio):
        self.out_red_ratio = ratio 
    
    def set_out_blue_ratio(self, ratio):
        self.out_blue_ratio = ratio

    def set_in_red_ratio(self, ratio):
        self.in_red_ratio = ratio 
    
    def set_in_blue_ratio(self, ratio):
        self.in_blue_ratio = ratio

    def set_out_blue(self, number):
        self.out_blue = number

    def set_out_red(self, number):
        self.out_red = number

    def set_in_blue(self, number):
        self.in_blue = number

    def set_in_red(self, number):
        self.in_red = number

    def set_x_to_give(self, phi):
        if (self.out_blue == 0) and (self.out_red) == 0:
            self.x_to_give = 0.0
        elif self.out_red_ratio <= phi:
            self.x_to_give = (1 - phi) / self.out_blue
        elif self.out_blue_ratio < (1 - phi):
            self.x_to_give = phi / self.out_red

    def set_excess_to_red(self, phi):
        if self.out_red_ratio < phi:
            numerator = (1 - phi) * self.out_red
            if self.out_blue == 0:
                fraction = 0
            else:
                fraction = numerator / self.out_blue
            self.excess_to_red = phi - fraction

    def set_excess_to_blue(self, phi):
        if self.out_blue_ratio < (1 - phi):
            numerator = phi * self.out_blue
            if self.out_red == 0:
                fraction = 0
            else:
                fraction = numerator / self.out_red
            self.excess_to_blue = (1 - phi) - fraction

    def set_excess_to_individuals(self, red_nodes, blue_nodes):
        if self.out_blue != 0:
            self.excess_to_blue_individual = self.excess_to_blue / self.out_blue
        else:
            self.excess_to_blue_individual = self.excess_to_blue / blue_nodes
        if self.excess_to_red_individual != 0:
            self.excess_to_red_individual = self.excess_to_red / self.out_red
        else:
            self.excess_to_red_individual = self.excess_to_red / red_nodes

    def set_importance_in_community(self, value):
        self.importance_in_community = value

class Graph():
    def __init__(self):
        self.no_of_nodes = self.get_no_of_nodes()
        self.nodes = np.array([Node() for i in range(self.no_of_nodes)])
        self.node_communities = np.array([0 for i in range(self.no_of_nodes)], dtype=int)
        self.delta_red = np.zeros(self.no_of_nodes)
        self.delta_blue = np.zeros(self.no_of_nodes)
        self.set_node_communitites()
        self.set_node_infos()
        self.transition_matrix = np.zeros((self.no_of_nodes, self.no_of_nodes))
    
    def get_no_of_nodes(self):
        n = 0
        with open("out_graph.txt", "r") as file_one:
            n = int(file_one.readline())

        return n
    
    def set_node_communitites(self):
        with open("out_community.txt", "r") as file_one:
            file_one.readline()
            for line in file_one:
                node_id = int(line.split()[0])
                node_cat = int(line.split()[1])
                self.node_communities[node_id] = node_cat

    def get_community_of_node(self, node):
        return self.node_communities[node]

    def set_node_infos(self):
        with open("out_graph.txt", "r") as file_one:
            file_one.readline()
            for line in file_one:
                node_source = int(line.split()[0])
                node_target = int(line.split()[1])
                self.nodes[node_source].out_neighbors.add(node_target)
                self.nodes[node_target].in_neighbors.add(node_source)
        
        node_id = 0
        for node in self.nodes:
            out_red = 0
            in_red = 0
            out_blue = 0
            in_blue = 0
            out_total = 0
            in_total = 0
            for out_nei in node.out_neighbors:
                out_total += 1
                if self.get_community_of_node(out_nei) == 1:
                    out_red += 1
                else:
                    out_blue +=1
            if out_total != 0:
                red_ratio = out_red / out_total
                blue_ratio = out_blue / out_total
            else:
                red_ratio = blue_ratio = 0
            node.set_out_blue(out_blue)
            node.set_out_red(out_red)
            node.set_out_red_ratio(red_ratio)
            node.set_out_blue_ratio(blue_ratio)
            for in_nei in node.in_neighbors:
                in_total += 1
                if self.get_community_of_node(in_nei) == 1:
                    in_red += 1
                else:
                    in_blue +=1
            if in_total != 0:
                red_ratio = in_red / in_total
                blue_ratio = in_blue / in_total
            else:
                red_ratio = blue_ratio = 0
            node.set_in_blue(out_blue)
            node.set_in_red(out_red)
            node.set_in_red_ratio(red_ratio)
            node.set_in_blue_ratio(blue_ratio)
            node.set_excess_to_individuals(np.sum(self.node_communities), np.sum(1 - self.node_communities))
            node.id = node_id
            node_id += 1

    def set_excess_deltas(self):
        for i in range(self.no_of_nodes):
            self.delta_red[i] = self.nodes[i].excess_to_red
            self.delta_blue[i] = self.nodes[i].excess_to_blue

    def set_transition_matrix(self):
        for node in range(self.no_of_nodes):
            for neighbor in self.nodes[node].out_neighbors:
                self.transition_matrix[node][neighbor] = self.nodes[node].x_to_give

    def get_jump_vector(self, phi):
        jump_vector = np.zeros(g.no_of_nodes)
        red_nodes = np.sum(g.node_communities)
        blue_nodes = g.no_of_nodes - red_nodes
        for i in range(g.no_of_nodes):
            if g.get_community_of_node(i) == 1:
                jump_vector[i] = phi / red_nodes
            else:
                jump_vector[i] = (1 - phi) / blue_nodes

        return jump_vector

    def set_importance_in_communities(self, pagerank_v):
        pgrnk_per_com = np.zeros(2)
        for i in range(self.no_of_nodes):
            com = g.get_community_of_node(i)
            pgrnk_per_com[com] += pagerank_v[i]

        for i in range(self.no_of_nodes):
            com = g.get_community_of_node(i)
            value = pagerank_v[i] / pgrnk_per_com[com]
            node.set_importance_in_community(value)

#-------------------------------------------------- MAIN ---------------------------------------------------#
# define phi.
if len(sys.argv) != 2:
    print("provide 1 arguments <ratio for protected category>")
else:
    PHI = float(sys.argv[1])

index = get_cat_indexes()

if PHI == 0:
    PHI = sum(index) / len(index)
print("phi: ", PHI)

def plots():
    # Init infos.
    g = Graph()
    g.set_node_communitites()
    g.set_node_infos()
    for node in g.nodes:
        node.set_x_to_give(PHI)
        node.set_excess_to_red(PHI)
        node.set_excess_to_blue(PHI)
    g.set_excess_deltas()

    # Load rank vectors in arrays.
    algorithms = ["pagerank", "lfprn", "lfpru", "lfprp"]
    out_file = ["out_pagerank_pagerank.txt", "out_lfpr_n_pagerank.txt", "out_lfpr_u_pagerank.txt", "out_lfpr_p_pagerank.txt"]

    rank_vectors = dict()
    exc_pol_red = dict()
    exc_pol_blue = dict()
    delta_red = g.delta_red
    delta_blue = g.delta_blue

    # Init rank vectors arrays.
    for i in range(len(algorithms)):
        rank_vectors[algorithms[i]] = np.zeros(g.no_of_nodes)
        exc_pol_red[algorithms[i]] = np.zeros(g.no_of_nodes)
        exc_pol_blue[algorithms[i]] = np.zeros(g.no_of_nodes)
        with open(out_file[i], "r") as file_one:
            j = 0
            for line in file_one:
                rank_vectors[algorithms[i]][j] = float(line)
                j += 1

    # Init importance in community for lfprp.
    g.set_importance_in_communities(rank_vectors["pagerank"])

    # Init excess policies.
    for i in range(g.no_of_nodes):
        #exc_pol_red["lfprn"][i] = g.nodes[i].excess_to_red_individual
        #exc_pol_blue["lfprn"][i] = g.nodes[i].excess_to_blue_individual
        if g.get_community_of_node(i) == 0:
            exc_pol_red["lfpru"][i] = 0
            exc_pol_red["lfprp"][i] = 0
            exc_pol_blue["lfpru"][i] = 1 / np.sum(1 - g.node_communities)
            exc_pol_blue["lfprp"][i] = g.nodes[i].importance_in_community
        else:
            exc_pol_red["lfpru"][i] = 1 / np.sum(g.node_communities)
            exc_pol_red["lfprp"][i] = g.nodes[i].importance_in_community
            exc_pol_blue["lfpru"][i] = 0
            exc_pol_blue["lfprp"][i] = 0

    # Init actual delta-policies.
    act_delta_red = dict()
    act_delta_blue = dict()
    act_exc_pol_red = dict()
    act_exc_pol_blue = dict()
    algorithms.remove("pagerank")
    for algorithm in algorithms:
        act_delta_red[algorithm] = np.multiply(delta_red, rank_vectors[algorithm])
        act_delta_blue[algorithm] = np.multiply(delta_blue, rank_vectors[algorithm])
        red_delta = np.dot(delta_red, rank_vectors[algorithm])
        blue_delta = np.dot(delta_blue, rank_vectors[algorithm])
        act_exc_pol_red[algorithm] = red_delta * exc_pol_red[algorithm] 
        act_exc_pol_blue[algorithm] = blue_delta * exc_pol_blue[algorithm] 

    # Plot distributions of delta, act_delta, exc_pol and act_exc_pol.
    algorithms.remove("lfprn")
    for algorithm in algorithms:
        # Act deltas
        fig = plt.figure(figsize=(10.,10.))
        plt.title("Value Deltas Distribution %s" %algorithm)
        values_min = min(np.amin(act_delta_blue[algorithm]), np.amin(act_delta_red[algorithm]))
        values_max = max(np.amax(act_delta_blue[algorithm]), np.amax(act_delta_red[algorithm]))
        lngth = (values_max - values_min) / no_of_bins
        plt.xticks(np.arange(values_min, values_max + lngth, lngth))
        plt.hist([act_delta_blue[algorithm], act_delta_red[algorithm]], color= ["b", "r"], bins= no_of_bins, weights= [np.ones(g.no_of_nodes) / g.no_of_nodes, np.ones(g.no_of_nodes) / g.no_of_nodes])
        plt.axvline(1 / g.no_of_nodes, label="Fair Ratio", linewidth = 1, color = "k", ls="--")
        plt.savefig("out_act_delta_distribution_%s.pdf" %algorithm)
        plt.savefig("out_act_delta_distribution_%s.png" %algorithm)
        plt.close()

        # Act exc_pol
        fig = plt.figure(figsize=(10.,10.))
        plt.title("Value Excess Policy Distribution %s" %algorithm)
        values_min = min(np.amin(act_exc_pol_blue[algorithm]), np.amin(act_exc_pol_red[algorithm]))
        values_max = max(np.amax(act_exc_pol_blue[algorithm]), np.amax(act_exc_pol_red[algorithm]))
        lngth = (values_max - values_min) / no_of_bins
        plt.xticks(np.arange(values_min, values_max + lngth, lngth))
        plt.hist([act_delta_blue[algorithm], act_delta_red[algorithm]], color= ["b", "r"], bins= no_of_bins, weights= [np.ones(g.no_of_nodes) / g.no_of_nodes, np.ones(g.no_of_nodes) / g.no_of_nodes])
        plt.axvline(1 / g.no_of_nodes, label="Fair Ratio", linewidth = 1, color = "k", ls="--")
        plt.savefig("out_act_exc_pol_distribution_%s.pdf" %algorithm)
        plt.savefig("out_act_exc_pol_distribution_%s.png" %algorithm)
        plt.close()

    algorithms.append("lfprn")
    # Deltas distributions.
    fig = plt.figure(figsize=(10.,10.))
    plt.title("delta Distribution")
    values_min = min(np.amin(delta_blue), np.amin(delta_blue))
    values_max = max(np.amax(delta_blue), np.amax(delta_blue))
    lngth = (values_max - values_min) / no_of_bins
    plt.xticks(np.arange(values_min, values_max + lngth, lngth))
    plt.hist([delta_blue, delta_red], color= ["b", "r"], bins= no_of_bins, align="left", weights= [np.ones(g.no_of_nodes) / g.no_of_nodes, np.ones(g.no_of_nodes) / g.no_of_nodes], rwidth=0.5)
    plt.axvline(1 / g.no_of_nodes, label="Fair Ratio", linewidth = 1, color = "k", ls="--")
    plt.savefig("delta_distribution.pdf")
    plt.savefig("deltal_distribution.png")
    plt.close()

    # Bar plots for algorithms.
    algorithms.append("pagerank")

    for algorithm in algorithms:
        index = np.argsort(-rank_vectors[algorithm])
        delta_red_temp = delta_red[index]
        delta_blue_temp = delta_blue[index]
        

        fig = plt.figure(figsize=(10.,10.))
        fig.suptitle("Delta to algorithms")
        plt.title(algorithm)
        plt.xlabel("%s ordered" %algorithm)
        plt.ylabel("delta value")
        plt.plot(np.arange(1,g.no_of_nodes + 1, 1), delta_blue_temp, ",b")
        plt.plot(np.arange(1, g.no_of_nodes + 1, 1), delta_red_temp, ",r")
        plt.savefig("delta_to_%s.pdf" %algorithm)
        plt.savefig("delta_to_%s.png" %algorithm)

def top_dif_analysis():
    # Init infos.
    g = Graph()
    g.set_node_communitites()
    g.set_node_infos()
    for node in g.nodes:
        node.set_x_to_give(PHI)
        node.set_excess_to_red(PHI)
        node.set_excess_to_blue(PHI)
    g.set_excess_deltas()

    with open("out_graph.txt", "r") as file_one:
        no_of_nodes = int(file_one.readline())

    # Load rank vectors in arrays.
    algorithms = ["pagerank", "lfprn", "lfpru", "lfprp"]
    out_file = ["out_pagerank_pagerank.txt", "out_lfpr_n_pagerank.txt", "out_lfpr_u_pagerank.txt", "out_lfpr_p_pagerank.txt"]

    rank_vectors = dict()
    sort_index = dict()
    dif_vectors = dict()

    # Init rank vectors arrays.
    for i in range(len(algorithms)):
        rank_vectors[algorithms[i]] = np.zeros(no_of_nodes)
        with open(out_file[i], "r") as file_one:
            j = 0
            for line in file_one:
                rank_vectors[algorithms[i]][j] = float(line)
                j += 1

    algorithms.remove("pagerank")
    pgrnk_index = np.argsort(-rank_vectors["pagerank"])
    for algo in algorithms:
        dif_vectors[algo] = rank_vectors[algo] - rank_vectors["pagerank"]
        k_min = np.argsort(dif_vectors[algo])[:10]
        k_max = np.argsort(-dif_vectors[algo])[:10]
        with open("out_%s_value_dif_weighted.txt" %algo, "w") as file_one:
            file_one.write("node\tdiff\tcom\tin_red_ratio\tout_red_ratio\texc_to_red\texc_to_blue\tavr_in_nei_out_red_ratio\tbest_in_nei_pgrnk_pos\n")
            for i in k_min:
                avrg_red_ratio = 0.0
                in_nei_pgrnk = 0.0
                pgrnk_pos = g.no_of_nodes
                for nghbr in g.nodes[i].in_neighbors:
                    in_nei_pgrnk += rank_vectors["pagerank"][nghbr]
                    #avrg_red_ratio += g.nodes[nghbr].out_red_ratio
                    avrg_red_ratio += g.nodes[nghbr].out_red_ratio * rank_vectors["pagerank"][nghbr]
                    pos = np.where(pgrnk_index == nghbr)[0][0]
                    if pos < pgrnk_pos:
                        pgrnk_pos = pos
                if len(g.nodes[i].in_neighbors) != 0:
                    #avrg_red_ratio = avrg_red_ratio / len(g.nodes[i].in_neighbors)
                    avrg_red_ratio = avrg_red_ratio / in_nei_pgrnk
                file_one.write("%d\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%d\n" %(i, dif_vectors[algo][i], g.get_community_of_node(i), g.nodes[i].in_red_ratio, g.nodes[i].out_red_ratio,
                                            g.nodes[i].excess_to_red, g.nodes[i].excess_to_blue, avrg_red_ratio, pgrnk_pos))
            for i in k_max:
                avrg_red_ratio = 0.0
                in_nei_pgrnk = 0.0
                pgrnk_pos = g.no_of_nodes
                for nghbr in g.nodes[i].in_neighbors:
                    in_nei_pgrnk += rank_vectors["pagerank"][nghbr]
                    #avrg_red_ratio += g.nodes[nghbr].out_red_ratio
                    avrg_red_ratio += g.nodes[nghbr].out_red_ratio * rank_vectors["pagerank"][nghbr]
                    pos = np.where(pgrnk_index == nghbr)[0][0]
                    if pos < pgrnk_pos:
                        pgrnk_pos = pos
                if len(g.nodes[i].in_neighbors) != 0:
                    #avrg_red_ratio = avrg_red_ratio / len(g.nodes[i].in_neighbors)
                    avrg_red_ratio = avrg_red_ratio / in_nei_pgrnk
                file_one.write("%d\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%d\n" %(i, dif_vectors[algo][i], g.get_community_of_node(i), g.nodes[i].in_red_ratio, g.nodes[i].out_red_ratio,
                                            g.nodes[i].excess_to_red, g.nodes[i].excess_to_blue, avrg_red_ratio, pgrnk_pos))
    algorithms.append("pagerank")

    # Init sort indexes.
    for algo in algorithms:
        sort_index[algo] = np.argsort(-rank_vectors[algo])

    # Find difference in positions.
    algorithms.remove("pagerank")
    for algo in algorithms:
        pos_vectors = np.zeros(no_of_nodes, dtype=int)
        for i in range(no_of_nodes):
            dif_vectors[algo][sort_index["pagerank"][i]] += i
            dif_vectors[algo][sort_index[algo][i]] -= i
    """
    k_min = dict()
    k_max = dict()

    # Find 10 min and max diff, positions.
    for algo in algorithms:
        min_index = np.argsort(dif_vectors[algo]) 
        max_index = np.argsort(-dif_vectors[algo])
        k_min[algo] = min_index[:10]
        k_max[algo] = max_index[:10]

    # Write in files.
    for algo in algorithms:
        with open("out_%s_rank_dif.txt" %algo, "w") as file_one:
            file_one.write("node \tdiff \tcom \tin_red_ratio \tout_red_ratio \texc_to_red \texc_to_blue \tavr_in_nei_out_red_ratio \tbest_in_nei_pgrnk_pos \n")
            for i in k_min[algo]:
                avrg_red_ratio = 0.0
                pgrnk_pos = g.no_of_nodes
                for nghbr in g.nodes[i].in_neighbors:
                    avrg_red_ratio += g.nodes[nghbr].out_red_ratio
                    pos = np.where(pgrnk_index == nghbr)[0][0]
                    if pos < pgrnk_pos:
                        pgrnk_pos = pos
                if len(g.nodes[i].in_neighbors) != 0:
                    avrg_red_ratio = avrg_red_ratio / len(g.nodes[i].in_neighbors)
                file_one.write("%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\n" %(i, dif_vectors[algo][i], g.get_community_of_node(i), g.nodes[i].in_red_ratio, g.nodes[i].out_red_ratio,
                                            g.nodes[i].excess_to_red, g.nodes[i].excess_to_blue, avrg_red_ratio, pgrnk_pos))
            for i in k_max[algo]:
                avrg_red_ratio = 0.0
                pgrnk_pos = g.no_of_nodes
                for nghbr in g.nodes[i].in_neighbors:
                    avrg_red_ratio += g.nodes[nghbr].out_red_ratio
                    pos = np.where(pgrnk_index == nghbr)[0][0]
                    if pos < pgrnk_pos:
                        pgrnk_pos = pos
                if len(g.nodes[i].in_neighbors) != 0:
                    avrg_red_ratio = avrg_red_ratio / len(g.nodes[i].in_neighbors)
                file_one.write("%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\n" %(i, dif_vectors[algo][i], g.get_community_of_node(i), g.nodes[i].in_red_ratio, g.nodes[i].out_red_ratio,
                                            g.nodes[i].excess_to_red, g.nodes[i].excess_to_blue, avrg_red_ratio, pgrnk_pos))
    """
top_dif_analysis()
