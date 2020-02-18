#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "graph.hpp"

graph::graph(const std::string &graph_filename, const std::string &com_filename)
{
	nedges = 0;
	load_num_nodes(graph_filename);
	load_communities(com_filename);
	load_graph(graph_filename);
}

/* graph::~graph() */
/* { */
/* 	if (nodes) */
/* 		delete [] nodes; */
/* 	if (communities) */
/* 		delete [] communities; */
/* 	if (attributes) */
/* 		delete [] attributes; */
/* } */

void graph::load_graph(const std::string &graph_filename)
{
	std::ifstream infile(graph_filename);

	infile >> nnodes;

	for (int node = 0; node < nnodes; ++node)
	{
		nodes[node].in_neighbors_per_community.resize(ncommunities);
		nodes[node].out_neighbors_per_community.resize(ncommunities);
	}

	int node1, node2;
	while (infile >> node1 >> node2)
	{
		add_edge(node1, node2);
	}

	infile.close();
}

void graph::load_num_nodes(const std::string &graph_filename)
{
	std::ifstream infile(graph_filename);
	infile >> nnodes;
	nodes = new node_t [nnodes];
	infile.close();
}

void graph::load_communities(const std::string &com_filename)
{
	std::ifstream infile(com_filename);

	infile >> ncommunities;

	communities = new community_t [ncommunities];

	for (int i = 0; i < ncommunities; ++i)
		communities[i].community_id = i;

	int node, community;
	while (infile >> node >> community)
	{
		nodes[node].community = community;
		communities[community].nodes.push_back(node);
	}

	infile.close();

	// give default community percentage
	comm_percentage.resize(ncommunities);
	for (int comm = 0; comm < ncommunities; ++comm)
		comm_percentage[comm] = communities[comm].nodes.size() / (double)nnodes;
}

void graph::load_attributes(const std::string &attribute_filename)
{
	std::ifstream infile(attribute_filename);

	infile >> nattributes;

	attributes = new attribute_t[nattributes];

	for (int i = 0; i < nattributes; ++i)
		attributes[i].attribute_id = i;

	std::string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		int node, attribute;

		iss >> node;
		while (iss >> attribute)
		{
			attributes[attribute].nodes.push_back(node);
		}
	}

	infile.close();
}

void graph::load_community_percentage(const std::string &percentage_filename)
{
	std::ifstream infile(percentage_filename);

	int community;
	double percentage;
	while (infile >> community >> percentage)
	{
		comm_percentage[community] = percentage;
	}

	infile.close();

	// check for correctness
	double sum = 0.0;
	for (const auto &percentage : comm_percentage)
		sum += percentage;
	if ((sum < 1-10e-6) || (sum > 1+10e-6))
		std::cerr << "[WARN] Sum of community percentage should be 1.0, but it is "
			<< sum << std::endl;
}

double graph::get_community_percentage(const int community) const
{
	return comm_percentage[community];
}

void graph::add_edge(const int src_node, const int dest_node)
{
	nodes[dest_node].in_neighbors.push_back(src_node);
	++nodes[src_node].out_neighbors_per_community[nodes[dest_node].community];
	++nodes[dest_node].in_neighbors_per_community[nodes[src_node].community];
	++nodes[src_node].out_degree;
	++nedges;
}

const std::vector<int> &graph::get_in_neighbors(int node_id) const
{
	return nodes[node_id].in_neighbors;
}

int graph::get_in_degree(const int node_id) const
{
	return nodes[node_id].in_neighbors.size();
}

int graph::get_out_degree(const int node_id) const
{
	return nodes[node_id].out_degree;
}

int graph::get_community(const int node) const
{
	return nodes[node].community;
}

int graph::count_in_neighbors_with_community(const int node_id, const int target_community)
{
	return nodes[node_id].in_neighbors_per_community[target_community];
}

int graph::count_out_neighbors_with_community(const int node_id, const int target_community)
{
	return nodes[node_id].out_neighbors_per_community[target_community];
}

std::vector<double> graph::get_pagerank_per_community(pagerank_v pagerankv) const
{
	std::vector<double> pagerank_per_community(ncommunities);

	for (int node = 0; node < nnodes; ++node) {
		double pagerank = pagerankv[node].pagerank;
		int community = nodes[node].community;
		pagerank_per_community[community] += pagerank;
	}

	return pagerank_per_community;
}

community_t *graph::get_communities()
{
	return communities;
}

int graph::get_community_size(const int community_id)
{
	return communities[community_id].nodes.size();
}

std::vector<int> &graph::get_nodes_with_attribute(const int attribute_id)
{
	return attributes[attribute_id].nodes;
}
