""" Gradient based optimization for the excess problem.
It solves the other constractions by projection. Not very efficient.
It also assumes that solution domain is convex so it is based only in
good initial point.

Because of the wrong assumptionof convexity we can not quarranty that
the way we cope with constraint is proper.

TODO:
    i. Make independent of convexity.
    ii. Think of possible ways to cope with  contstraints.
"""
import numpy as np 
import sys
import torch as torch
import math
import time

#----From Opt----

def uniformPR(M, gamma = 0.15):
    n = M.shape[0]
    d = np.array(np.reciprocal(M.sum(axis = 1).T))
    d = d.flatten()
    D = np.diag(np.array(d))
    P = D.dot(M) # up to here xorrect - M must have float elements
    Q = gamma*np.linalg.inv((np.eye(n) - (1-gamma)*P))
    Q = Q.T# up to here also correct
    u = (np.ones(n)/(1.0*n)).T
    p = Q.dot(u) # also correct
    return p # Returns pagerank vector and Q matrix

def create_adj_matrix(filename = "out_graph.txt"):
    n = 0
    with open(filename, "r") as file_one:
        n = int(file_one.readline())

    M = np.zeros([n,n])

    with open(filename, "r") as file_one:
        file_one.readline()
        for line in file_one:
            edge = line.split()
            M[int(edge[0])][int(edge[1])] = 1.
            
    index = [False for i in range(n)]
    with open("out_community.txt", "r") as file_one:
        file_one.readline()
        for line in file_one:
            info = line.split()
            if int(info[1]) == 1:
                index[int(info[0])] = True

    j = 0
    for i in range(n):
        if not M[i].any():
            j += 1
            M[i] = [1. for i in M[i]]
    print("%d vectors without out neighbors" %j)

    return M, index

#------New------

class Node():
    def __init__(self):
        self.out_neighbors = set()
        self.in_neighbors = set()
        self.x_to_give = 0.0
        self.excess_to_red = 0.0
        self.excess_to_blue = 0.0
        self.out_red = 0
        self.out_blue = 0
        self.out_red_ratio = 0.0
        self.out_blue_ratio = 0.0
        self.import_in_com = 0.0 # Importance in community.

    def set_node_id(self, number):
        self.id = number

    def set_out_red_ratio(self, ratio):
        self.out_red_ratio = ratio 
    
    def set_out_blue_ratio(self, ratio):
        self.out_blue_ratio = ratio

    def set_out_blue(self, number):
        self.out_blue = number

    def set_out_red(self, number):
        self.out_red = number

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
        self.index = np.zeros(self.no_of_nodes)
    
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
        
        for node in self.nodes:
            out_red = 0
            out_blue = 0
            out_total = 0
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
        
        red_sum = 0.0
        blue_sum = 0.0
        with open("out_pagerank_pagerank.txt", "r") as file_one:
            for i in range(self.no_of_nodes):
                value = float(file_one.readline())
                self.nodes[i].import_in_com = value
                if self.get_community_of_node(i) == 1:
                    red_sum += value 
                else:
                    blue_sum += value 
        for i in range(self.no_of_nodes):
            if self.get_community_of_node(i) == 1:
                self.nodes[i].import_in_com = self.nodes[i].import_in_com / red_sum
            else:
                self.nodes[i].import_in_com = self.nodes[i].import_in_com / blue_sum

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

    def set_index(self, index):
        self.index = index

class Gradient_descent():
    def __init__(self, PL, delta_red, delta_blue, pagerank_vector, jump_vector, community_indexes, index):
        self.dimension = delta_red.size
        self.red_nodes_num = np.sum(community_indexes)
        self.blue_nodes_num = np.sum(1 - community_indexes)
        self.pagerank = torch.tensor(pagerank_vector)
        self.jump_vector = torch.tensor(jump_vector)
        self.PL = torch.tensor(PL)
        self.delta_red = torch.tensor(delta_red)
        self.delta_blue = torch.tensor(delta_blue)
        self.community_indexes = torch.tensor(community_indexes.astype(float))
        self.best_point = torch.tensor(np.zeros(self.dimension))
        self.current_x_red = torch.tensor(np.array([1 / self.red_nodes_num for i in range(self.red_nodes_num)])).clone().detach()
        self.current_x_blue = torch.tensor(np.array([1 / self.blue_nodes_num for i in range(self.blue_nodes_num)])).clone().detach()
        self.best_loss_value = 1.0
        self.current_loss_value = 1.0
        self.index = index
        #self.direction = torch.tensor(np.zeros(self.dimension))

    def init_current_points(self, index):
        #print("Init points")
        exc_pol = np.zeros(self.dimension)
        with open("out_excess_sensitive_policy_v.txt", "r") as file_one:
            for i in range(self.dimension):
                exc_pol[i] = float(file_one.readline())
        #print(exc_pol[:10])
        exc_pol = exc_pol[index]
        for i in range(self.blue_nodes_num):
            self.current_x_blue[i] = exc_pol[i]
        for i in range(self.red_nodes_num):
            self.current_x_red[i] = exc_pol[self.blue_nodes_num + i]

    def init_points_by_vec(self, vec, index):
        vec = vec[index]
        for i in range(self.blue_nodes_num):
            self.current_x_blue[i] = vec[i]
        for i in range(self.red_nodes_num):
            self.current_x_red[i] = vec[self.blue_nodes_num + i]

    def get_dist_from_pagrank(self, x_red, x_blue):
        # Define other parameters than pl, deltas, pagerank.
        id_matrix = torch.tensor(np.identity(self.dimension))
        gama = 0.15
        # Define the function.
        excess_array = (self.PL + torch.cat((self.delta_blue.reshape(self.dimension, 1) * x_blue, self.delta_red.reshape(self.dimension, 1) * x_red), 1))
        y = gama * torch.matmul(self.jump_vector, torch.inverse(id_matrix - ((1 - gama) * excess_array)))

        # Define the loss.
        distance = torch.norm((y - self.pagerank))
        
        return distance 

    def get_loss_at(self, x_red, x_blue):
        global loss_calc
        loss_calc += 1
        # Define other parameters than pl, deltas, pagerank.
        id_matrix = torch.tensor(np.identity(self.dimension))
        gama = 0.15
        # Soft constraint multiplier.
        l = 2

        # Define the function.
        excess_array = (self.PL + torch.cat((self.delta_blue.reshape(self.dimension, 1) * x_blue, self.delta_red.reshape(self.dimension, 1) * x_red), 1))
        y = gama * torch.matmul(self.jump_vector, torch.inverse(id_matrix - ((1 - gama) * excess_array)))
        # Define the loss.
        # distance = torch.sum((y - self.pagerank)**2)
        distance = torch.dist(y, self.pagerank, p=2) 
        loss = distance ** 2

        return loss

    def get_gradient(self):
        x_red = self.current_x_red.clone().detach().requires_grad_(True)
        x_blue= self.current_x_blue.clone().detach().requires_grad_(True)

        loss = self.get_loss_at(x_red, x_blue)
        
        # Compute gradients
        loss.backward()

        # Return gradient.
        return x_red.grad, x_blue.grad
        
    def line_search(self):
        global loss_time
        global line_time
        global dir_time
        global start_time
        global loss_calc
        global converge_file
        global time_file
        self.current_loss_value = self.get_loss_at(self.current_x_red, self.current_x_blue)

        dir_start = time.time()
        red_direction, blue_direction = self.get_gradient()
        dir_stop = time.time()
        dir_elapsed = dir_stop - dir_start
        dir_time += dir_elapsed
        red_direction = - red_direction
        blue_direction = - blue_direction
        
        line_start = time.time()

        step = 1
        temp_red = self.current_x_red + step * red_direction
        temp_blue = self.current_x_blue + step * blue_direction
        loss_start = time.time()
        temp_loss = self.get_loss_at(temp_red, temp_blue)
        loss_stop = time.time()
        loss_elapsed = loss_stop - loss_start
        loss_time += loss_elapsed
        
        # Guaranties no <0 or >1 coordinates.
        valide_values = True
        for i in temp_red:
            if i < 0 or i > 1:
                valide_values = False
        for i in temp_blue:
            if i < 0 or i > 1:
                valide_values = False

        while (not valide_values):
            step = step / 2
            temp_red = self.current_x_red + step * red_direction
            temp_blue = self.current_x_blue + step * blue_direction
            if (step < (10 ** (-30))):
                time_file = open("out_proj_gradient_timing.txt", "a")
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_file.write("%f\t%f(%d)\t%f\t%f\n" %(elapsed_time, loss_time, loss_calc, dir_time, line_time))
                time_file.close()
                converge_file.close()
                self.save_results(self.index)
                sys.exit("step < 10 ^ (-30)")
            valide_values = True
            for i in temp_red:
                if i < 0 or i > 1:
                    valide_values = False
            for i in temp_blue:
                if i < 0 or i > 1:
                    valide_values = False

        # Guarranties that sum or Red, Blue == 1 by projection.
        red_sum = torch.sum(temp_red)
        blue_sum = torch.sum(temp_blue)
        temp_red = (1 / red_sum) * temp_red
        temp_blue = (1 / blue_sum) * temp_blue

        # Start All over again
        red_direction = temp_red - self.current_x_red
        blue_direction = temp_blue - self.current_x_blue
        step = 1
        loss_start = time.time()
        temp_loss = self.get_loss_at(temp_red, temp_blue)
        loss_stop = time.time()
        loss_elapsed = loss_stop - loss_start
        loss_time += loss_elapsed
 
        #print(self.current_loss_value)
        while (temp_loss > self.current_loss_value):
            step = step / 2 # try * 9 / 10.
            temp_red = self.current_x_red + step * red_direction
            temp_blue = self.current_x_blue + step * blue_direction
            loss_start = time.time()
            temp_loss = self.get_loss_at(temp_red, temp_blue)
            loss_stop = time.time()
            loss_elapsed = loss_stop - loss_start
            loss_time += loss_elapsed
            #print(temp_loss)
            if (step < (10 ** (-30))):
                time_file = open("out_proj_gradient_timing.txt", "a")
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_file.write("%f\t%f(%d)\t%f\t%f\n" %(elapsed_time, loss_time, loss_calc, dir_time, line_time))
                time_file.close()
                converge_file.close()
                self.save_results(self.index)
                sys.exit("step < 10 ^ (-30)--!")
        
            
        self.current_x_red = temp_red
        self.current_x_blue = temp_blue
        #temp_loss = self.get_loss_at(self.current_x_red, self.current_x_blue)
        if temp_loss <= self.current_loss_value:
            self.current_loss_value = temp_loss
        else:
            time_file = open("out_proj_gradient_timing.txt", "a")
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_file.write("%f\t%f(%d)\t%f\t%f\n" %(elapsed_time, loss_time, loss_calc, dir_time, line_time))
            time_file.close()
            converge_file.close()
            sys.exit("New loss value greater than previous but this should not have happend")
        #print("test print|| Loss_value: ", self.current_loss_value, " Step: ", step, " real loss: ", self.get_dist_from_pagrank(self.current_x_red, self.current_x_blue) / 0.0011)

        line_end = time.time()
        line_time += line_end - line_start

        return red_direction, blue_direction

    def optimize(self):
        global converge_file
        # self.init_current_points() || For different initialization than uniform.
        self.current_loss_value = self.get_loss_at(self.current_x_red, self.current_x_blue)
        gradient = torch.cat(self.get_gradient())
        #print("test print|| Loss_value: ", self.current_loss_value, " real loss: ", self.get_dist_from_pagrank(self.current_x_red, self.current_x_blue))
        
        itera = 0
        local_start_time = time.time()
        now = local_start_time
        elapsed_time = now - local_start_time
        converge_file.write("%d\t%f\t%f\n" %(itera, elapsed_time, self.current_loss_value))
        while(torch.norm(gradient) > (10 ** (-4)) and itera < 200):
            gradient = torch.cat(self.line_search())
            itera += 1
            print("iter: ", itera, ", loss: ", self.current_loss_value)
            if itera % 10 == 0:
                self.save_results(self.index)
            now = time.time()
            elapsed_time = now - local_start_time
            converge_file.write("%d\t%f\t%f\n" %(itera, elapsed_time, self.current_loss_value))
        
        self.save_results(self.index)
        print(torch.sum(self.current_x_red), torch.sum(self.current_x_blue))
    
    def save_results(self, index):
        # Define other parameters than pl, deltas, pagerank.
        id_matrix = torch.tensor(np.identity(self.dimension))
        gama = 0.15
        # Define the function.
        excess_array = (self.PL + torch.cat((self.delta_blue.reshape(self.dimension, 1) * self.current_x_blue, self.delta_red.reshape(self.dimension, 1) * self.current_x_red), 1))
        y = gama * torch.matmul(self.jump_vector, torch.inverse(id_matrix - ((1 - gama) * excess_array)))

        y = y.data.numpy()
        y_indx = np.zeros(self.dimension)
        exc_policy = torch.cat((self.current_x_blue, self.current_x_red)).data.numpy()
        exc_indx = np.zeros(self.dimension)
        for i in range(index.size):
            y_indx[index[i]] = y[i]
            exc_indx[index[i]] = exc_policy[i]

        

        #print("Sum of opt excess pagerank vector: ", np.sum(y))
        #print("Sum of excess vector: ", torch.sum(self.current_x_blue) + torch.sum(self.current_x_red))
        with open("out_proj_excess_sensitive_pagerank.txt", "w") as file_one:
            for i in range(y_indx.size):
                file_one.write("%f\n" %y_indx[i])
        with open("out_proj_excess_sensitive_policy_v.txt", "w") as file_one:
            for i in range(exc_indx.size):
                file_one.write("%f\n" %exc_indx[i])

#-------------------------------------------------- MAIN ---------------------------------------------------#
# Log files.
loss_time = 0.
loss_calc = 0
line_time = 0.
dir_time = 0.
time_file = open("out_proj_gradient_timing.txt", "w")
time_file.write("total\t\tloss\t\tdir\t\tline_search\n")
time_file.close()
converge_file = open("out_proj_gradient_converge.txt", "w")
converge_file.write("iter\ttime\t\tloss\n")

# define phi.
if len(sys.argv) != 2:
    print("provide 1 arguments <ratio for protected category>")
else:
    PHI = float(sys.argv[1])

# Start timer.
start_time = time.time() # In seconds.
# Get pagerank.
M,index = create_adj_matrix()
pagerank_vector = uniformPR(M)

if PHI == 0:
    PHI = sum(index) / len(index)
print("phi: ", PHI)

# Get Gradient descent parameters.
g = Graph()
g.set_node_communitites()
g.set_node_infos()
for node in g.nodes:
    node.set_x_to_give(PHI)
    node.set_excess_to_red(PHI)
    node.set_excess_to_blue(PHI)
g.set_excess_deltas()
g.set_transition_matrix()

#Classify nodes. Blue firsts, Red seconds.
index = np.argsort(g.node_communities)
g.set_index(index)

# Initialize optimizer.
for i in range(g.no_of_nodes):
    g.transition_matrix[i] = g.transition_matrix[i][index]
opt = Gradient_descent(g.transition_matrix[index], g.delta_red[index], g.delta_blue[index], pagerank_vector[index],
                       g.get_jump_vector(PHI)[index], g.node_communities[index], g.index)
# Init From file.
# opt.init_current_points(index).
# Start from Proportional.
vec = np.zeros(g.no_of_nodes)
for i in range(g.no_of_nodes):
    vec[i] = g.nodes[i].import_in_com
opt.init_points_by_vec(vec, index)


opt.optimize()
end_time = time.time()
converge_file.close()
opt.save_results(index)
time_file = open("out_proj_gradient_timing.txt", "a")
elapsed_time = end_time - start_time
time_file.write("%f\t%f(%d)\t%f\t%f\n" %(elapsed_time, loss_time, loss_calc, dir_time, line_time))
time_file.close()
print("normally terminated")






