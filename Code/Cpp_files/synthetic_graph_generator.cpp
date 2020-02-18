#include <vector>
#include <random>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <algorithm>

#define MAX_TRIES 1000 /* maximum tries to add an edge */

enum community_type {NON_PROTECTED = 0, PROTECTED};

typedef struct {
	std::vector<int> neighbors;
	community_type community;
} node_t;

static node_t *nodes;
static int total_degree = 0; // sum of degrees of all nodes

static double get_rand()
{
	return rand() / ((double)RAND_MAX + 1);
}

static void assign_community_to_node(const int node, const double protected_prob)
{
	if (get_rand() < protected_prob)
		nodes[node].community = community_type::PROTECTED;
	else
		nodes[node].community = community_type::NON_PROTECTED;
}


static void init_graph(const int init_nnodes, const double protected_prob)
{
	for (int node = 0; node < init_nnodes; ++node)
	{
		assign_community_to_node(node, protected_prob);
		for (int other_node = 0; other_node < node; ++other_node)
		{
			nodes[node].neighbors.push_back(other_node);
			nodes[other_node].neighbors.push_back(node);
			total_degree += 2;
		}
	}

	// make sure there is at least one initial node of each community
	nodes[0].community = community_type::NON_PROTECTED;
	nodes[1].community = community_type::PROTECTED;
}

static void add_rest_nodes(const int init_nnodes, const int nnodes, const int edges_per_node,
		const double protected_prob, const double homophily_0, const double homophily_1)
{
	std::vector<int> tmp_degree(nnodes);

	for (int node = init_nnodes; node < nnodes; node++) // add rest of the nodes
	{
		assign_community_to_node(node, protected_prob);
		double homophily = (nodes[node].community == 0) ? homophily_0 : homophily_1;

		int skipped_edges = 0;
		std::fill(tmp_degree.begin(), tmp_degree.end(), 0);

		for (int edge = 0; edge < edges_per_node; ++edge)
		{
			int tries = 0;
insert_edge:
			if (++tries > MAX_TRIES)
			{
				++skipped_edges;
				std::cerr << "Skipping edge " << edge << " for node " << node <<
					" after " << MAX_TRIES << " tries" << std::endl;
				continue;
			}

			double prob = get_rand(), sum = 0.0;
			int node_to_connect = 0;
			while ((sum += (nodes[node_to_connect].neighbors.size()-tmp_degree[node_to_connect]) / (double)total_degree) < prob)
				++node_to_connect;

			if (tmp_degree[node_to_connect] > 0)
				goto insert_edge; // we already added this edge

			prob = get_rand();
			if ((nodes[node].community == nodes[node_to_connect].community) && (prob < homophily))
				goto insert_edge;
			else if ((nodes[node].community != nodes[node_to_connect].community) && (prob >= homophily))
				goto insert_edge;

			nodes[node].neighbors.push_back(node_to_connect);
			++tmp_degree[node];
			nodes[node_to_connect].neighbors.push_back(node);
			++tmp_degree[node_to_connect];
		}
		total_degree += 2 * (edges_per_node - skipped_edges);

		if (skipped_edges == edges_per_node)
		{
			// We were not able to insert any edge; insert one at random to
			// make sure we have a connected graph of the given size at the end.
			double prob = get_rand(), sum = 0.0;
			int node_to_connect = 0;
			while ((sum += 1.0 / node) < prob)
				++node_to_connect;
			std::cout << "Manually connecting nodes " << node << " and " << node_to_connect << std::endl;
			nodes[node].neighbors.push_back(node_to_connect);
			nodes[node_to_connect].neighbors.push_back(node);
			total_degree += 2;
		}
	}
}

static void save_graph(const std::string &graph_filename, const std::string &community_filename,
		const int nnodes)
{
	std::ofstream out_graph(graph_filename);
	std::ofstream out_community(community_filename);

	out_graph << nnodes << std::endl;
	for (int node = 0; node < nnodes; ++node)
		for (const auto &neighbor : nodes[node].neighbors)
			out_graph << node << " " << neighbor << std::endl;

	out_community << 2 << std::endl;
	for (int node = 0; node < nnodes; ++node)
		out_community << node << " " << nodes[node].community << std::endl;

	out_graph.close();
	out_community.close();
}

int main(int argc, char *argv[])
{
	srand(time(nullptr));
	if (argc != 7)
	{
		std::cerr << "Usage: " << argv[0] << " <homophily_0> <homophily_1> <protected_prob> "
			"<number_of_nodes> <edges_per_node> <init_number_of_nodes>" << std::endl;
		return 1;
	}

	// 0.0 perfect homophily (all edges are between vertices of the same catogory) to 1.0 (no homophily)
	double homophily_0 = atof(argv[1]); // homophily for community 0
	double homophily_1 = atof(argv[2]); // homophily for community 1
	double protected_prob = atof(argv[3]); // probability that the new node is of the proteted community
	int nnodes = atoi(argv[4]); // total number of nodes the graph must have in the end
	int edges_per_node = atoi(argv[5]); // number of edges each node should create (except the initial ones)
	int init_nnodes = atoi(argv[6]); // number of initial nodes

	if (nnodes < init_nnodes)
	{
		std::cerr << "Number of nodes must be greater or equal to the number of initial nodes." << std::endl;
		exit(EXIT_FAILURE);
	}
	if (init_nnodes < 2)
	{
		std::cerr << "We must have at least 2 initial nodes." << std::endl;
		exit(EXIT_FAILURE);
	}

	nodes = new node_t[nnodes];
	init_graph(init_nnodes, protected_prob);
	add_rest_nodes(init_nnodes, nnodes, edges_per_node, protected_prob, homophily_0, homophily_1);
	save_graph("out_graph.txt", "out_community.txt", nnodes);
}
