#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <omp.h>

#include "graph.hpp"
#include "pagerank.hpp"

int main()
{
	graph g("out_graph.txt", "out_community.txt");
	pagerank_algorithms algs(g);
	const int nnodes = g.get_num_nodes();
	const int ncommunities = g.get_num_communities();

	std::ofstream outfile("out_person_all_nodes.txt");
	outfile << std::fixed;
	outfile << std::setprecision(9);
	outfile << "node_id\tnode_pagerank";
	for (int comm = 0; comm < ncommunities; ++comm)
		outfile << "\tsum_pagerank_" << comm;
	outfile << std::endl;

	double *sum_pagerank = new double[ncommunities];
	for (int node = 0; node < nnodes; ++node)
	{
		algs.set_personalization_type(personalization_t::NODE_PERSONALIZATION, node);
		pagerank_v pagerankv = algs.get_pagerank();

		std::fill_n(sum_pagerank, ncommunities, 0);
		#pragma omp parallel for firstprivate(nnodes) reduction(+:sum_pagerank[:ncommunities])
		for (int i = 0; i < nnodes; ++i)
			sum_pagerank[g.get_community(i)] += pagerankv[i].pagerank;

		outfile << node << '\t' << pagerankv[node].pagerank;
		for (int comm = 0; comm < ncommunities; ++comm)
			outfile << '\t' << sum_pagerank[comm];
		outfile << std::endl;
	}
	delete [] sum_pagerank;
}
