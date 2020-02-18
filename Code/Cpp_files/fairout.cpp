#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <unordered_map>

#include "graph.hpp"
#include "pagerank.hpp"


static std::vector<double> get_fairout(graph &g)
{
	const int nnodes = g.get_num_nodes();
	std::vector<double> fairout(nnodes);

	for (int node = 0; node < nnodes; ++node)
		fairout[node] = (g.get_out_degree(node) == 0) ?
			g.get_community_size(1) / (double)g.get_num_nodes() :
			g.count_out_neighbors_with_community(node, 1) / (double)g.get_out_degree(node);

	return fairout;
}


static std::vector<double> get_in_neighbor_fairout(graph &g, std::vector<double> &fairout)
{
	const int nnodes = g.get_num_nodes();
	std::vector<double> in_neighbor_fairout(nnodes);

	for (int node = 0; node < nnodes; ++node)
	{
		double sum_neigh_fairout = 0.0;
		for (const auto &neigh : g.get_in_neighbors(node))
			sum_neigh_fairout += fairout[neigh];
		in_neighbor_fairout[node] = (g.get_in_degree(node) == 0) ?
			g.get_community_size(1) / (double)g.get_num_nodes() :
			sum_neigh_fairout / (double)g.get_in_degree(node);
	}

	return in_neighbor_fairout;
}


static std::vector<double> get_in_neighbor_weighted_fairout(graph &g,
		std::vector<double> &fairout, std::unordered_map<int, double> &pagerank)
{
	const int nnodes = g.get_num_nodes();
	std::vector<double> in_neighbors_weighted_fairout(nnodes);

	for (int node = 0; node < nnodes; ++node)
	{
		double sum_weighted_neigh_fairout = 0.0, sum_pagerank = 0.0;
		for (const auto &neigh : g.get_in_neighbors(node))
		{
			sum_weighted_neigh_fairout += pagerank[neigh] * fairout[neigh];
			sum_pagerank += pagerank[neigh];
		}
		in_neighbors_weighted_fairout[node] = (sum_pagerank == 0.0) ?
			g.get_community_size(1) / (double)g.get_num_nodes() :
			sum_weighted_neigh_fairout / sum_pagerank;
	}

	return in_neighbors_weighted_fairout;
}


static void get_average(graph &g, std::vector<double> &fairout,
		double &total, double &comm_0, double &comm_1)
{
	const int nnodes = g.get_num_nodes();
	total = comm_0 = comm_1 = 0.0;

	for (int node = 0; node < nnodes; ++node)
	{
		total += fairout[node];
		if (g.get_community(node) == 0)
			comm_0 += fairout[node];
		else
			comm_1 += fairout[node];
	}

	total /= g.get_num_nodes();
	comm_0 /= g.get_community_size(0);
	comm_1 /= g.get_community_size(1);
}


static void get_weighted_average(graph &g, std::vector<double> &fairout, std::unordered_map<int, double> &pagerank,
		double &total, double &comm_0, double &comm_1)
{
	const int nnodes = g.get_num_nodes();
	double sum_pg_total = 0.0, sum_pg_comm_0 = 0.0, sum_pg_comm_1 = 0.0;
	total = comm_0 = comm_1 = 0.0;

	for (int node = 0; node < nnodes; ++node)
	{
		const double node_pagerank = pagerank[node];
		total += node_pagerank * fairout[node];
		sum_pg_total += node_pagerank;
		if (g.get_community(node) == 0)
		{
			comm_0 += node_pagerank * fairout[node];
			sum_pg_comm_0 += node_pagerank;
		}
		else
		{
			comm_1 += node_pagerank * fairout[node];
			sum_pg_comm_1 += node_pagerank;
		}
	}

	total /= sum_pg_total;
	comm_0 /= sum_pg_comm_0;
	comm_1 /= sum_pg_comm_1;
}


int main()
{
	graph g("out_graph.txt", "out_community.txt");
	pagerank_algorithms algs(g);
	algs.set_personalization_type(personalization_t::NO_PERSONALIZATION, 0);
	pagerank_v pagerankv = algs.get_pagerank();

	std::unordered_map<int, double> pagerank; // key = node_id, value = pagerank
	for (const auto &node : pagerankv)
		pagerank.insert({node.node_id, node.pagerank});

	std::vector<double> fairout = get_fairout(g);
	std::vector<double> in_neighbor_fairout = get_in_neighbor_fairout(g, fairout);
	std::vector<double> in_neighbor_weighted_fairout = get_in_neighbor_weighted_fairout(g, fairout, pagerank);

	std::ofstream outfile("out_fairout.txt");
	outfile << std::fixed;
	outfile << std::setprecision(9);
	double total, comm_0, comm_1;

	get_average(g, fairout, total, comm_0, comm_1);
	outfile << "Average fairout for all nodes  : " << total << std::endl;
	outfile << "Average fairout for community 0: " << comm_0 << std::endl;
	outfile << "Average fairout for community 1: " << comm_1 << std::endl << std::endl;

	get_weighted_average(g, fairout, pagerank, total, comm_0, comm_1);
	outfile << "Weighted average fairout for all nodes  : " << total << std::endl;
	outfile << "Weighted average fairout for community 0: " << comm_0 << std::endl;
	outfile << "Weighted average fairout for community 1: " << comm_1 << std::endl << std::endl;

	get_average(g, in_neighbor_fairout, total, comm_0, comm_1);
	outfile << "Average in-neighbor fairout for all nodes  : " << total << std::endl;
	outfile << "Average in-neighbor fairout for community 0: " << comm_0 << std::endl;
	outfile << "Average in-neighbor fairout for community 1: " << comm_1 << std::endl << std::endl;

	get_weighted_average(g, in_neighbor_weighted_fairout, pagerank, total, comm_0, comm_1);
	outfile << "Weighted average in-neighbor weighted fairout for all nodes  : " << total << std::endl;
	outfile << "Weighted average in-neighbor weighted fairout for community 0: " << comm_0 << std::endl;
	outfile << "Weighted average in-neighbor weighted fairout for community 1: " << comm_1 << std::endl << std::endl;

	return 0;
}
