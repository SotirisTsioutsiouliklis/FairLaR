#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <cmath>
#include <cassert>
#include <omp.h>
#include "pagerank.hpp"
#include <stdlib.h>

pagerank_algorithms::pagerank_algorithms(graph &g) : g(g), is_cache_valid(false)
{
	// nothing to do here
}

pagerank_v pagerank_algorithms::get_pagerank(const double C, const double eps, const int max_iter)
{
	// initialize
	const unsigned int nnodes = g.get_num_nodes();
	pagerank_v pagerankv(nnodes);
	unsigned int i, node;
	#pragma omp parallel for firstprivate(nnodes)
	for (i = 0; i < nnodes; ++i) {
		pagerankv[i].node_id = i;
		pagerankv[i].pagerank = 1.0 / nnodes;
	}

	// compute pagerank for each node
	std::vector<double> tmp_pagerank(nnodes);
	std::vector<double> tmp_pagerank_jump(nnodes); // for personalization
	int iter = 0;
	for (; iter < max_iter; ++iter) {
		double sum = 0.0;
		#pragma omp parallel for firstprivate(nnodes) reduction(+:sum)
		for (node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] = 0.0;
			for (const int &neighbor : g.get_in_neighbors(node)) {
				int neigh_degree = g.get_out_degree(neighbor);
				if (neigh_degree > 0)
					tmp_pagerank[node] += pagerankv[neighbor].pagerank / neigh_degree;
			}
			tmp_pagerank[node] *= C;
			sum += tmp_pagerank[node];
		}

		// re-insert "leaked" pagerank
		double diff = 0.0, new_val;
		const double leaked = (C - sum) / nnodes;
		if (personalization_type == personalization_t::NO_PERSONALIZATION)
			compute_pagerank_no_personalization_vector(tmp_pagerank_jump, 1 - C);
		else
			compute_personalization_vector(tmp_pagerank_jump, 1 - C);
		#pragma omp parallel for firstprivate(nnodes, tmp_pagerank, tmp_pagerank_jump) private(new_val) reduction(+:diff)
		for (node = 0; node < nnodes; ++node) {
			new_val = tmp_pagerank[node] + leaked + tmp_pagerank_jump[node];
			diff += std::fabs(new_val - pagerankv[node].pagerank);
			pagerankv[node].pagerank = new_val;
		}
		if (diff < eps) break;
	}

	if (iter == max_iter)
		std::cerr << "[WARN]: Pagerank algorithm reached " << max_iter << " iterations." << std::endl;
	cached_pagerank = pagerankv;
	is_cache_valid = true;
	return pagerankv;
}

pagerank_v pagerank_algorithms::get_global_uniform_fair_pagerank(const double C,
		const bool use_cached, const double eps, const int max_iter)
{
	typedef struct {
		double owed_pagerank; // sum of pagerank owed by the community
		int nbankrupt_nodes; // number of nodes with 0 pagerank that cannot give more
	} owed_t;

	pagerank_v pagerankv;
	if (use_cached && is_cache_valid)
		pagerankv = cached_pagerank;
	else
		pagerankv = get_pagerank(C, eps, max_iter);

	// total sum of pagerank for each community
	std::vector<double> pagerank_per_community = g.get_pagerank_per_community(pagerankv);

	// vector of nodes that make up each community
	community_t *community = g.get_communities();

	// compute how much pagerank each community should give(+) / take(-) from other
	// communities
	std::vector<double> pagerank_diff(g.get_num_communities(), 0.0);
	for (unsigned int i = 0; i < pagerank_diff.size(); ++i)
		// pagerank each community has minus pagerank each community should have
		pagerank_diff[i] = pagerank_per_community[i] - g.get_community_percentage(i);

	// redistribute pagerank making sure that no node ends with negative pagerank
	std::vector<owed_t> owed_communities(g.get_num_communities());
	for (unsigned int comm = 0; comm < pagerank_diff.size(); ++comm)
		for (const auto &node : community[comm].nodes)
		{
			pagerankv[node].pagerank += (double)-pagerank_diff[comm] / (double)g.get_community_size(comm);
			if (pagerankv[node].pagerank < 0.0)
			{
				owed_communities[comm].owed_pagerank -= pagerankv[node].pagerank; // - because its negative
				++owed_communities[comm].nbankrupt_nodes;
				pagerankv[node].pagerank = 0.0;
			}
		}

	// deal with bankrupted communities
	for (unsigned int comm = 0; comm < owed_communities.size(); ++comm)
	{
		while (owed_communities[comm].owed_pagerank > 0.0)
		{
			double owed_pagerank = owed_communities[comm].owed_pagerank;
			int working_nodes = community[comm].nodes.size() - owed_communities[comm].nbankrupt_nodes;
			owed_communities[comm].owed_pagerank = 0.0;

			for (const auto &node : community[comm].nodes)
			{
				if (pagerankv[node].pagerank > 0.0)
				{
					pagerankv[node].pagerank -= owed_pagerank / (double)working_nodes;
					if (pagerankv[node].pagerank < 0.0)
					{
						owed_communities[comm].owed_pagerank -= pagerankv[node].pagerank; // - because its negative
						++owed_communities[comm].nbankrupt_nodes;
						pagerankv[node].pagerank = 0.0;
					}
				}
			}
		}
	}

	return pagerankv;
}

pagerank_v pagerank_algorithms::get_global_proportional_fair_pagerank(const double C,
		const bool use_cached, const double eps, const int max_iter)
{
	pagerank_v pagerankv;
	if (use_cached && is_cache_valid)
		pagerankv = cached_pagerank;
	else
		pagerankv = get_pagerank(C, eps, max_iter);

	// total sum of pagerank for each community
	std::vector<double> pagerank_per_community = g.get_pagerank_per_community(pagerankv);

	// redistribute pagerank
	int nnodes = g.get_num_nodes();
	#pragma omp parallel for firstprivate(nnodes)
	for (int node = 0; node < nnodes; ++node) {
		int comm = g.get_community(node);
		pagerankv[node].pagerank = pagerankv[node].pagerank / pagerank_per_community[comm] *
			g.get_community_percentage(comm);
	}

	return pagerankv;
}

pagerank_v pagerank_algorithms::get_step_uniform_fair_pagerank(const double C,
		const bool use_cached, const double eps, const int max_iter)
{
	return get_step_fair_pagerank(C, use_cached, eps, max_iter, true);
}

pagerank_v pagerank_algorithms::get_step_proportional_fair_pagerank(const double C,
		const bool use_cached, const double eps, const int max_iter)
{
	return get_step_fair_pagerank(C, use_cached, eps, max_iter, false);
}

pagerank_v pagerank_algorithms::get_custom_step_fair_pagerank(std::vector<double> &custom_excess, const double C, const bool use_cached,
			const double eps, const int max_iter, const bool run_uniform)
{
	// initialize
	const unsigned int nnodes = g.get_num_nodes();
	const int ncommunities = g.get_num_communities();
	std::vector<node_info_t> node_info(nnodes);
	initialize_node_info(node_info, use_cached, C, eps, max_iter);
	pagerank_v pagerankv(nnodes);
	unsigned int i, node;
	#pragma omp parallel for firstprivate(nnodes)
	for (i = 0; i < nnodes; ++i) {
		pagerankv[i].node_id = i;
		pagerankv[i].pagerank = 1.0 / nnodes;
	}

	// compute pagerank for each node.
	std::vector<double> tmp_pagerank(nnodes);
	std::vector<double> tmp_pagerank_jump(nnodes); // for personalization.
	double *community_pagerank = new double[ncommunities]; // sum of pagerank each community should get.
	int iter = 0;
	for (; iter < max_iter; ++iter) {
		std::fill_n(community_pagerank, ncommunities, 0);
		#pragma omp parallel for firstprivate(nnodes) reduction(+:community_pagerank[:ncommunities])
		for (node = 0; node < nnodes; ++node) {
			// take pagerank from neighbors.
			tmp_pagerank[node] = 0.0;
			for (const int &neigh : g.get_in_neighbors(node))
				tmp_pagerank[node] += pagerankv[neigh].pagerank *
					(node_info[neigh].node_percentage);

			// give node's extra pagerank to the communities buckets.
			for (int comm = 0; comm < ncommunities; ++comm)
				community_pagerank[comm] += pagerankv[node].pagerank * node_info[node].community_percentage[comm];
		}

		// re-insert "leaked" pagerank.
		double diff = 0.0, sum = 0.0, new_val;
		#pragma omp parallel for reduction(+:sum)
		for (unsigned int i = 0; i < tmp_pagerank.size(); ++i) {
			const int comm = g.get_community(i);
			const double comm_pagerank = community_pagerank[comm] * custom_excess[i];
			tmp_pagerank[i] = C * (tmp_pagerank[i] + comm_pagerank);
			sum += tmp_pagerank[i];
		}
		const double leaked = (C - sum); // for all nodes.
		if (!run_uniform && personalization_type == personalization_t::NO_PERSONALIZATION)
			compute_step_proportional_no_personalization_vector(tmp_pagerank_jump, 1 - C, node_info);
		else
			compute_personalization_vector(tmp_pagerank_jump, 1 - C);
		#pragma omp parallel for firstprivate(nnodes, tmp_pagerank, tmp_pagerank_jump) private(new_val) reduction(+:diff)
		for (node = 0; node < nnodes; ++node) {
			const int comm = g.get_community(node);
			double my_leaked;
			if (run_uniform)
				my_leaked = leaked * g.get_community_percentage(comm) / g.get_community_size(comm);
			else
				my_leaked = leaked * g.get_community_percentage(comm) * node_info[node].importance_in_community;
			new_val = tmp_pagerank[node] + my_leaked + tmp_pagerank_jump[node];
			diff += std::fabs(new_val - pagerankv[node].pagerank);
			pagerankv[node].pagerank = new_val;
		}
		if (diff < eps) break;
	}
	
	if (iter == max_iter)
		std::cerr << "[WARN]: Step " << ((run_uniform) ? "uniform" : "proportional") <<
			" fair Pagerank algorithm reached " << max_iter << " iterations." << std::endl;
	delete [] community_pagerank;
	return pagerankv;
}

pagerank_v pagerank_algorithms::get_step_fair_pagerank(const double C,
		const bool use_cached, const double eps, const int max_iter, const bool run_uniform)
{
	// initialize
	const unsigned int nnodes = g.get_num_nodes();
	const int ncommunities = g.get_num_communities();
	std::vector<node_info_t> node_info(nnodes);
	initialize_node_info(node_info, use_cached, C, eps, max_iter);
	pagerank_v pagerankv(nnodes);
	unsigned int i, node;
	#pragma omp parallel for firstprivate(nnodes)
	for (i = 0; i < nnodes; ++i) {
		pagerankv[i].node_id = i;
		pagerankv[i].pagerank = 1.0 / nnodes;
	}

	// compute pagerank for each node
	std::vector<double> tmp_pagerank(nnodes);
	std::vector<double> tmp_pagerank_jump(nnodes); // for personalization
	double *community_pagerank = new double[ncommunities]; // sum of pagerank each community should get
	int iter = 0;
	for (; iter < max_iter; ++iter) {
		std::fill_n(community_pagerank, ncommunities, 0);
		#pragma omp parallel for firstprivate(nnodes) reduction(+:community_pagerank[:ncommunities])
		for (node = 0; node < nnodes; ++node) {
			// take pagerank from neighbors
			tmp_pagerank[node] = 0.0;
			for (const int &neigh : g.get_in_neighbors(node))
				tmp_pagerank[node] += pagerankv[neigh].pagerank *
					(node_info[neigh].node_percentage);

			// give node's extra pagerank to the communities buckets
			for (int comm = 0; comm < ncommunities; ++comm)
				community_pagerank[comm] += pagerankv[node].pagerank * node_info[node].community_percentage[comm];
		}

		// re-insert "leaked" pagerank
		double diff = 0.0, sum = 0.0, new_val;
		#pragma omp parallel for reduction(+:sum)
		for (unsigned int i = 0; i < tmp_pagerank.size(); ++i) {
			const int comm = g.get_community(i);
			const double comm_pagerank = (run_uniform) ?
				community_pagerank[comm] / (double)g.get_community_size(comm):
				community_pagerank[comm] * node_info[i].importance_in_community;
			tmp_pagerank[i] = C * (tmp_pagerank[i] + comm_pagerank);
			sum += tmp_pagerank[i];
		}
		const double leaked = (C - sum); // for all nodes
		/*if (!run_uniform && personalization_type == personalization_t::NO_PERSONALIZATION)
			compute_step_proportional_no_personalization_vector(tmp_pagerank_jump, 1 - C, node_info);
		else
			compute_personalization_vector(tmp_pagerank_jump, 1 - C);
		*/
		compute_personalization_vector(tmp_pagerank_jump, 1 - C);
		#pragma omp parallel for firstprivate(nnodes, tmp_pagerank, tmp_pagerank_jump) private(new_val) reduction(+:diff)
		for (node = 0; node < nnodes; ++node) {
			const int comm = g.get_community(node);
			double my_leaked;
			if (run_uniform)
				my_leaked = leaked * g.get_community_percentage(comm) / g.get_community_size(comm);
			else
				my_leaked = leaked * g.get_community_percentage(comm) * node_info[node].importance_in_community;
			new_val = tmp_pagerank[node] + my_leaked + tmp_pagerank_jump[node];
			diff += std::fabs(new_val - pagerankv[node].pagerank);
			pagerankv[node].pagerank = new_val;
		}
		if (diff < eps) break;
	}

	if (iter == max_iter)
		std::cerr << "[WARN]: Step " << ((run_uniform) ? "uniform" : "proportional") <<
			" fair Pagerank algorithm reached " << max_iter << " iterations." << std::endl;
	delete [] community_pagerank;
	return pagerankv;
}

std::vector<double> pagerank_algorithms::get_proportional_excess_vector(const double C,
		const bool use_cached, const double eps, const int max_iter)
		{
	const unsigned int nnodes = g.get_num_nodes();
	std::vector<node_info_t> node_info(nnodes);
	initialize_node_info(node_info, use_cached, C, eps, max_iter);
	std::vector<double> proportional_excess(nnodes);
	for (unsigned int i = 0; i < nnodes; i++) {
		proportional_excess[i] = node_info[i].importance_in_community;
	}

	return proportional_excess;
}

void pagerank_algorithms::initialize_node_info(std::vector<node_info_t> &node_info, const bool use_cached_pagerank,
		const double C, const double eps, const int max_iter)
{
	pagerank_v pagerankv;
	if (use_cached_pagerank && is_cache_valid)
		pagerankv = cached_pagerank;
	else
		pagerankv = get_pagerank(C, eps, max_iter);

	assert(g.get_num_communities() == 2); // FIXME not implemented for the general case

	// total sum of pagerank for each community
	std::vector<double> pagerank_per_community = g.get_pagerank_per_community(pagerankv);

	const int nnodes = g.get_num_nodes();
	const int ncommunities = g.get_num_communities();

	// max percentage of pagerank each community can get
	std::vector<double> max_community_percentage(ncommunities);
	for (int comm = 0; comm < ncommunities; ++comm)
		max_community_percentage[comm] = g.get_community_percentage(comm);

	#pragma omp parallel for firstprivate(nnodes)
	for (int node = 0; node < nnodes; ++node) {
		int my_comm = g.get_community(node);
		node_info[node].importance_in_community = pagerankv[node].pagerank / pagerank_per_community[my_comm];
		node_info[node].community_percentage.resize(ncommunities);
		if (g.get_out_degree(node) == 0)
		{
			// node does not have any neighbors; community percentage must
			// be equal to max community percentage
			std::copy(max_community_percentage.begin(), max_community_percentage.end(),
					node_info[node].community_percentage.begin());
			continue;
		}
		int favorable_comm = 0;
		for (int comm = 0; comm < ncommunities; ++comm) {
			if ((g.count_out_neighbors_with_community(node, comm) / (double)g.get_out_degree(node)) >= g.get_community_percentage(comm))
				favorable_comm = comm;
		}
		node_info[node].node_percentage = g.get_community_percentage(favorable_comm) /
			(double)g.count_out_neighbors_with_community(node, favorable_comm);

		for (int comm = 0; comm < ncommunities; ++comm)
			if (comm == favorable_comm)
				node_info[node].community_percentage[comm] = 0.0;
			else
				node_info[node].community_percentage[comm] = 1.0 - (g.get_out_degree(node) * node_info[node].node_percentage);
	}
}

// For topk.
void pagerank_algorithms::initialize_node_info_topk(std::vector<node_topk> &node_info)
{
	const int nnodes = g.get_num_nodes();
	std::vector<int> out_nei(nnodes, 0); 
	std::vector<int> out_top_nei(nnodes, 0); 

	// Find pagerank for topk for every in topk node.
	for (int i = 0; i < nnodes; i++) {
		for (const int &neighbor : g.get_in_neighbors(i)) {
			out_nei[neighbor]++;
			if (node_info[i].is_at_topk) {
				node_info[neighbor].is_in_topk = true;
				out_top_nei[neighbor]++;
				node_info[neighbor].out_topk_neighbors_per_community[g.get_community(i)]++;
				}
		}
	}
	for (int i = 0; i < nnodes; i++) {
		double pgrnk_topk = (out_nei[i] != 0) ? out_top_nei[i] / (double)out_nei[i] : 0;
		if (node_info[i].is_in_topk) {
			node_info[i].favour_community = (node_info[i].out_topk_neighbors_per_community[0] / (double)out_top_nei[i] >= g.get_community_percentage(0)) ? 0 : 1;
			node_info[i].pagerank_for_topk = pgrnk_topk * g.get_community_percentage(node_info[i].favour_community) / (double)node_info[i].out_topk_neighbors_per_community[node_info[i].favour_community];
			node_info[i].excess = pgrnk_topk - (node_info[i].pagerank_for_topk * out_top_nei[i]);
		}
	}
}

pagerank_v pagerank_algorithms::get_local_fair_pagerank(const double C, const double eps,
		const int max_iter)
{
	int nnodes = g.get_num_nodes();
	int ncommunities = g.get_num_communities();

	// initialize pagerank
	pagerank_v pagerankv(nnodes);
	#pragma omp parallel for firstprivate(nnodes)
	for (int i = 0; i < nnodes; ++i) {
		pagerankv[i].node_id = i;
		pagerankv[i].pagerank = 1.0 / nnodes;
	}

	// compute pagerank for each node
	std::vector<double> tmp_pagerank(nnodes);
	std::vector<double> tmp_pagerank_jump(nnodes); // for personalization
	double *community_pagerank = new double[ncommunities]; // sum of pagerank each community should get
	int iter = 0;
	for (; iter < max_iter; ++iter) {
		std::fill_n(community_pagerank, ncommunities, 0);
		#pragma omp parallel for firstprivate(nnodes) reduction(+:community_pagerank[:ncommunities])
		for (int node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] = 0.0;
			const int community = g.get_community(node);
			// take pagerank from neighbors
			for (const int &neighbor : g.get_in_neighbors(node)) {
				const int nsame_com = g.count_out_neighbors_with_community(neighbor, community);
				tmp_pagerank[node] += g.get_community_percentage(community) * pagerankv[neighbor].pagerank / (double)nsame_com;
			}
			// give percentage of pagerank to communities we have 0 neighbors to
			for (int comm = 0; comm < ncommunities; ++comm)
				if (g.count_out_neighbors_with_community(node, comm) == 0)
					community_pagerank[comm] += pagerankv[node].pagerank * g.get_community_percentage(comm);
		}

		// re-insert "leaked" pagerank
		double diff = 0.0, sum = 0.0, new_val;
		#pragma omp parallel for reduction(+:sum)
		for (unsigned int i = 0; i < tmp_pagerank.size(); ++i) {
			const int comm = g.get_community(i);
			tmp_pagerank[i] = C * (tmp_pagerank[i] + community_pagerank[comm] / (double)g.get_community_size(comm));
			sum += tmp_pagerank[i];
		}
		const double leaked = (C - sum); // for all nodes
		compute_personalization_vector(tmp_pagerank_jump, 1 - C);
		#pragma omp parallel for firstprivate(nnodes, tmp_pagerank, tmp_pagerank_jump) private(new_val) reduction(+:diff)
		for (int node = 0; node < nnodes; ++node) {
			const int comm = g.get_community(node);
			new_val = tmp_pagerank[node] + (leaked * g.get_community_percentage(comm) / g.get_community_size(comm)) + tmp_pagerank_jump[node];
			diff += fabs(new_val - pagerankv[node].pagerank);
			pagerankv[node].pagerank = new_val;
		}
		if (diff < eps) break;
	}

	if (iter == max_iter)
		std::cerr << "[WARN]: Local Fair Pagerank algorithm reached " << max_iter << " iterations." << std::endl;
	return pagerankv;
}

// LFPR_N top k ---- works for two communities only. Fix me.
pagerank_v pagerank_algorithms::get_lfprn_topk(const int k, const double C, const bool use_cached, const double eps,
		const int max_iter)
{
	// Take pagerank so as to know the importance of each node. 
	pagerank_v pure_pagerank;
	if (use_cached && is_cache_valid)
		pure_pagerank = cached_pagerank;
	else
		pure_pagerank = get_pagerank(C, eps, max_iter);

	int nnodes = g.get_num_nodes();
	int ncommunities = g.get_num_communities();

	// Sort pagerank vector so as to know the topk nodes.
	sort_pagerank_vector(pure_pagerank);
	// Separate into their category.
	std::vector<int> topk_red;
	std::vector<int> topk_blue;
	// Check if there is at least one node of each category at topk (personilization_extra). If not print the smaller fissible k.
	std::vector<bool> cat_at_topk(ncommunities, false); // Vector to check if there is at least one member of every community in topk.
	std::vector<node_topk> node_info_for_topk(nnodes); // Vector to store informations for topk algorithms.
	// Check of given K. cat_at_topk[i] == true if exists a member of category i in the topk by pagerank.
	for (int i = 0; i < k; i++)
	{
		int comm = g.get_community(pure_pagerank[i].node_id);
		cat_at_topk[comm] = true;
		node_info_for_topk[pure_pagerank[i].node_id].is_at_topk = true; // Flag for topk
		if (comm) {
			topk_red.push_back(pure_pagerank[i].node_id);
		} else {
			topk_blue.push_back(pure_pagerank[i].node_id);
		}
	}
	// K is valide if exists at least one of each category. Find if this is true.
	bool valide_k = true;
	for (auto i = cat_at_topk.begin(); i != cat_at_topk.end(); ++ i)
	{
		valide_k = valide_k && *i;
 	}
	// If given K is not fissible, find the smallest fissible K.
	if (!valide_k)
	{
		for (int i = k; i < nnodes; i++)
		{
			cat_at_topk[g.get_community(pure_pagerank[i].node_id)] = true;
			bool valide_k = true; // K is valide if exists at least one of each category.
			for (auto i = cat_at_topk.begin(); i != cat_at_topk.end(); ++ i) 
			{
				valide_k = valide_k && *i;
			}
			if (valide_k)
			{
				std::cout << "Not possible topk for K: " << k << ", Smallest possible K is: " << ++i;
				exit(0);
			}		
		}
	}

	// So k is fissible.
	// Init extra infos for topk.
	initialize_node_info_topk(node_info_for_topk); // Initialization is correct.

	// initialize pagerank.
	pagerank_v pagerankv(nnodes);
	for (int i = 0; i < nnodes; ++i) {
		pagerankv[i].node_id = i;
		pagerankv[i].pagerank = 1.0 / nnodes;
	}

	// compute pagerank for each node
	std::vector<double> tmp_pagerank(nnodes);
	std::vector<double> tmp_pagerank_jump(nnodes); // Jump vector.
	int iter = 0;
	for (; iter < max_iter; ++iter) {
		// Give pagerank to nodes.
		for (int node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] = 0.0;
			// Case topk nodes.
			if (node_info_for_topk[node].is_at_topk)
			{	
				const int comm = g.get_community(node);
				// Take pagerank from neighbors.
				for (const int &neighbor : g.get_in_neighbors(node))
				{	
					tmp_pagerank[node] += pagerankv[neighbor].pagerank * node_info_for_topk[neighbor].pagerank_for_topk;
					if (node_info_for_topk[neighbor].favour_community != comm) {
						tmp_pagerank[node] += pagerankv[neighbor].pagerank * node_info_for_topk[neighbor].excess / 
											  (double)node_info_for_topk[neighbor].out_topk_neighbors_per_community[comm];
					} 
				}
			}
			// Case non topk nodes.
			else
			{
				// take pagerank from neighbors
				for (const int &neighbor : g.get_in_neighbors(node)) {
					int neigh_degree = g.get_out_degree(neighbor);
					if (neigh_degree > 0) // Do we need it - It has at list one out neighbor the <node>.
						tmp_pagerank[node] += pagerankv[neighbor].pagerank / neigh_degree;
					// Test print. It seems correct.
					// std::cout << "node: " << node << " temp: " << tmp_pagerank[node] << std::endl;
				}
			}
		}
		// Give pagerank of in_topk Nodes without neighbors of one category to all topk Nodes of that category.
		for (int node = 0; node < nnodes; node++) {
			if (node_info_for_topk[node].is_in_topk && (node_info_for_topk[node].out_topk_neighbors_per_community[0] == 0)) {
				for (const int &i : topk_blue) {
					tmp_pagerank[i] += pagerankv[node].pagerank * node_info_for_topk[node].excess / (double)topk_blue.size();
				}
			} else if (node_info_for_topk[node].is_in_topk && (node_info_for_topk[node].out_topk_neighbors_per_community[1] == 0)) {
				for (const int &i : topk_red) {
					tmp_pagerank[i] += pagerankv[node].pagerank * node_info_for_topk[node].excess / (double)topk_red.size();
				}
			}
		}

		double sum = 0;
		for (int node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] *= C;
			sum += tmp_pagerank[node];
		}
		// re-insert "leaked" pagerank
		double diff = 0.0, new_val;
		const double leaked = C - sum;
		for (int node = 0; node < nnodes; ++node) {
			if (node_info_for_topk[node].is_at_topk) {
				int comm = g.get_community(node);
				int topk_same_com = 0;
				topk_same_com = (comm == 0) ? (int)topk_blue.size() : (int)topk_red.size();
				new_val = tmp_pagerank[node] + ((leaked + 0.15) * (k / (double)nnodes) * g.get_community_percentage(comm) / (double)topk_same_com);
			} else {
				new_val = tmp_pagerank[node] + ((leaked + 0.15) / (double)nnodes);
			}
			diff += std::fabs(new_val - pagerankv[node].pagerank);
			pagerankv[node].pagerank = new_val;
		}
		if (diff < eps) break;
	}
	
	// Test print.
	double sum = 0; 
	double phi = 0;
	for (int i : topk_blue) sum += pagerankv[i].pagerank; 
	for (int i : topk_red) {sum += pagerankv[i].pagerank; phi += pagerankv[i].pagerank;}
	std::cout << "community 0 topk: " << 1 - phi / (double)sum << std::endl;
	std::cout << "community 1 topk: " << phi / (double)sum << std::endl;
	

	if (iter == max_iter)
		std::cerr << "[WARN]: Local Fair Neighborhood Pagerank algorithm reached " << max_iter << " iterations." << std::endl;
	return pagerankv;
}

// LFPR_U for top k ----- works for two communities only.
pagerank_v pagerank_algorithms::get_lfpru_topk(const int k, const double C, const bool use_cached, const double eps,
		const int max_iter)
{
	// Take pagerank so as to know the importance of each node. 
	pagerank_v pure_pagerank;
	if (use_cached && is_cache_valid)
		pure_pagerank = cached_pagerank;
	else
		pure_pagerank = get_pagerank(C, eps, max_iter);

	int nnodes = g.get_num_nodes();
	int ncommunities = g.get_num_communities();

	// Sort pagerank vector so as to know the topk nodes.
	sort_pagerank_vector(pure_pagerank);
	// Separate into their category.
	std::vector<int> topk_red;
	std::vector<int> topk_blue;
	// Check if there is at least one node of each category at topk (personilization_extra). If not print the smaller fissible k.
	std::vector<bool> cat_at_topk(ncommunities, false); // Vector to check if there is at least one member of every community in topk.
	std::vector<node_topk> node_info_for_topk(nnodes); // Vector to store informations for topk algorithms.
	// Check of given K. cat_at_topk[i] == true if exists a member of category i in the topk by pagerank.
	for (int i = 0; i < k; i++)
	{
		int comm = g.get_community(pure_pagerank[i].node_id);
		cat_at_topk[comm] = true;
		node_info_for_topk[pure_pagerank[i].node_id].is_at_topk = true; // Flag for topk
		if (comm) {
			topk_red.push_back(pure_pagerank[i].node_id);
		} else {
			topk_blue.push_back(pure_pagerank[i].node_id);
		}
	}
	// K is valide if exists at least one of each category. Find if this is true.
	bool valide_k = true;
	for (auto i = cat_at_topk.begin(); i != cat_at_topk.end(); ++ i)
	{
		valide_k = valide_k && *i;
 	}
	// If given K is not fissible, find the smallest fissible K.
	if (!valide_k)
	{
		for (int i = k; i < nnodes; i++)
		{
			cat_at_topk[g.get_community(pure_pagerank[i].node_id)] = true;
			bool valide_k = true; // K is valide if exists at least one of each category.
			for (auto i = cat_at_topk.begin(); i != cat_at_topk.end(); ++ i) 
			{
				valide_k = valide_k && *i;
			}
			if (valide_k)
			{
				std::cout << "Not possible topk for K: " << k << ", Smallest possible K is: " << ++i;
				exit(0);
			}		
		}
	}

	// So k is fissible.
	// Init extra infos for topk.
	initialize_node_info_topk(node_info_for_topk); // Initialization is correct.

	// initialize pagerank.
	pagerank_v pagerankv(nnodes);
	for (int i = 0; i < nnodes; ++i) {
		pagerankv[i].node_id = i;
		pagerankv[i].pagerank = 1.0 / nnodes;
	}

	// compute pagerank for each node
	std::vector<double> tmp_pagerank(nnodes);
	std::vector<double> tmp_pagerank_jump(nnodes); // Jump vector.
	int iter = 0;
	for (; iter < max_iter; ++iter) {
		// Give pagerank to nodes.
		for (int node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] = 0.0;
			// Case topk nodes.
			if (node_info_for_topk[node].is_at_topk)
			{	
				const int comm = g.get_community(node);
				// Take pagerank from neighbors.
				for (const int &neighbor : g.get_in_neighbors(node)) {
					tmp_pagerank[node] += pagerankv[neighbor].pagerank * node_info_for_topk[neighbor].pagerank_for_topk;
				}
			}
			// Case non topk nodes.
			else
			{
				// take pagerank from neighbors
				for (const int &neighbor : g.get_in_neighbors(node)) {
					int neigh_degree = g.get_out_degree(neighbor);
					if (neigh_degree > 0) // Do we need it - It has at list one out neighbor the <node>.
						tmp_pagerank[node] += pagerankv[neighbor].pagerank / neigh_degree;
				}
			}
		}
		// Give excess of in_topk Nodes.
		for (int node = 0; node < nnodes; node++) {
			if (node_info_for_topk[node].is_in_topk && ((node_info_for_topk[node].out_topk_neighbors_per_community[0] == 0) || (node_info_for_topk[node].favour_community != 0))) {
				for (const int &i : topk_blue) {
					tmp_pagerank[i] += pagerankv[node].pagerank * node_info_for_topk[node].excess / (double)topk_blue.size();
				}
			} else if (node_info_for_topk[node].is_in_topk && ((node_info_for_topk[node].out_topk_neighbors_per_community[1] == 0) || (node_info_for_topk[node].favour_community != 1))) {
				for (const int &i : topk_red) {
					tmp_pagerank[i] += pagerankv[node].pagerank * node_info_for_topk[node].excess / (double)topk_red.size();
				}
			}
		}

		double sum = 0;
		for (int node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] *= C;
			sum += tmp_pagerank[node];
		}
		// re-insert "leaked" pagerank
		double diff = 0.0, new_val;
		const double leaked = C - sum;
		for (int node = 0; node < nnodes; ++node) {
			if (node_info_for_topk[node].is_at_topk) {
				int comm = g.get_community(node);
				int topk_same_com = 0;
				topk_same_com = (comm == 0) ? (int)topk_blue.size() : (int)topk_red.size();
				new_val = tmp_pagerank[node] + ((leaked + 0.15) * (k / (double)nnodes) * g.get_community_percentage(comm) / (double)topk_same_com);
			} else {
				new_val = tmp_pagerank[node] + ((leaked + 0.15) / (double)nnodes);
			}
			diff += std::fabs(new_val - pagerankv[node].pagerank);
			pagerankv[node].pagerank = new_val;
		}
		if (diff < eps) break;
	}
	
	// Test print.
	double sum = 0; 
	double phi = 0;
	for (int i : topk_blue) sum += pagerankv[i].pagerank; 
	for (int i : topk_red) {sum += pagerankv[i].pagerank; phi += pagerankv[i].pagerank;}
	std::cout << "community 0 topk: " << 1 - phi / (double)sum << std::endl;
	std::cout << "community 1 topk: " << phi / (double)sum << std::endl;
	

	if (iter == max_iter)
		std::cerr << "[WARN]: Local Fair Uniform Pagerank algorithm reached " << max_iter << " iterations." << std::endl;
	return pagerankv;
}

// LFPR_P for top k ----- works for two communities only.
pagerank_v pagerank_algorithms::get_lfprp_topk(const int k, const double C, const bool use_cached, const double eps,
		const int max_iter)
{
	// Take pagerank so as to know the importance of each node. 
	pagerank_v pure_pagerank;
	pagerank_v pure_pagerank_unsorted;
	if (use_cached && is_cache_valid)
		pure_pagerank = cached_pagerank;
	else
		pure_pagerank = get_pagerank(C, eps, max_iter);

	pure_pagerank_unsorted = pure_pagerank;
	int nnodes = g.get_num_nodes();
	int ncommunities = g.get_num_communities();

	// Sort pagerank vector so as to know the topk nodes.
	sort_pagerank_vector(pure_pagerank);

	// Separate into their category.
	std::vector<int> topk_red;
	std::vector<int> topk_blue;

	// Check if there is at least one node of each category at topk.
	// If not print the smaller fissible k.
	std::vector<bool> cat_at_topk(ncommunities, false); // Vector to check if there is at least one member of every community in topk.
	std::vector<node_topk> node_info_for_topk(nnodes); // Vector to store informations for nodes for topk algorithms.
	
	for (int i = 0; i < k; i++)
	{
		int comm = g.get_community(pure_pagerank[i].node_id);
		cat_at_topk[comm] = true;
		node_info_for_topk[pure_pagerank[i].node_id].is_at_topk = true; // Flag for topk
		if (comm) {
			topk_red.push_back(pure_pagerank[i].node_id);
		} else {
			topk_blue.push_back(pure_pagerank[i].node_id);
		}
	}

	// K is valide if exists at least one of each category. Find if this is true.
	bool valide_k = true;
	for (auto i = cat_at_topk.begin(); i != cat_at_topk.end(); ++ i)
	{
		valide_k = valide_k && *i;
 	}
	// If given K is not fissible, find the smallest fissible K.
	if (!valide_k)
	{
		for (int i = k; i < nnodes; i++)
		{
			cat_at_topk[g.get_community(pure_pagerank[i].node_id)] = true;
			bool valide_k = true; // K is valide if exists at least one of each category.
			for (auto i = cat_at_topk.begin(); i != cat_at_topk.end(); ++ i) 
			{
				valide_k = valide_k && *i;
			}
			if (valide_k)
			{
				std::cout << "Not possible topk for K: " << k << ", Smallest possible K is: " << ++i;
				exit(0);
			}		
		}
	}

	// So k is fissible.
	// Init extra infos for topk.
	initialize_node_info_topk(node_info_for_topk); // Initialization is correct.

	// Init importance in top_k community.
	double red_topk_pgrnk = 0;
	double blue_topk_pgrnk = 0;
	std::vector<int>::iterator node;

	for (int node : topk_red) {
		red_topk_pgrnk += pure_pagerank_unsorted[node].pagerank;
	}
	for (int node : topk_red) {
		node_info_for_topk[node].importance_in_topk_community = pure_pagerank_unsorted[node].pagerank / (double)red_topk_pgrnk;
	}

	for (int node : topk_blue) {
		blue_topk_pgrnk += pure_pagerank_unsorted[node].pagerank;
	}
	for (int node : topk_blue) {
		node_info_for_topk[node].importance_in_topk_community = pure_pagerank_unsorted[node].pagerank / (double)blue_topk_pgrnk;
	}
	
	// initialize pagerank.
	pagerank_v pagerankv(nnodes);
	for (int i = 0; i < nnodes; ++i) {
		pagerankv[i].node_id = i;
		pagerankv[i].pagerank = 1.0 / nnodes;
	}

	// compute pagerank for each node
	std::vector<double> tmp_pagerank(nnodes);
	std::vector<double> tmp_pagerank_jump(nnodes); // Jump vector.
	int iter = 0;
	for (; iter < max_iter; ++iter) {
		// Give pagerank to nodes.
		for (int node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] = 0.0;
			// Case topk nodes.
			if (node_info_for_topk[node].is_at_topk)
			{	
				const int comm = g.get_community(node);
				// Take pagerank from neighbors.
				for (const int &neighbor : g.get_in_neighbors(node)) {	
					tmp_pagerank[node] += pagerankv[neighbor].pagerank * node_info_for_topk[neighbor].pagerank_for_topk;
				} 
			}		
			// Case non topk nodes.
			else
			{
				// take pagerank from neighbors
				for (const int &neighbor : g.get_in_neighbors(node)) {
					int neigh_degree = g.get_out_degree(neighbor);
					if (neigh_degree > 0) // Do we need it - It has at list one out neighbor the <node>.
						tmp_pagerank[node] += pagerankv[neighbor].pagerank / neigh_degree;
				}
			}
		}
		// Give excess of in_topk Nodes.
		for (int node = 0; node < nnodes; node++) {
			if (node_info_for_topk[node].is_in_topk && ((node_info_for_topk[node].out_topk_neighbors_per_community[0] == 0) || (node_info_for_topk[node].favour_community != 0))) {
				for (const int &i : topk_blue) {
					tmp_pagerank[i] += pagerankv[node].pagerank * node_info_for_topk[node].excess * node_info_for_topk[i].importance_in_topk_community;
				}
			} else if (node_info_for_topk[node].is_in_topk && ((node_info_for_topk[node].out_topk_neighbors_per_community[1] == 0) || (node_info_for_topk[node].favour_community != 1))) {
				for (const int &i : topk_red) {
					tmp_pagerank[i] += pagerankv[node].pagerank * node_info_for_topk[node].excess * node_info_for_topk[i].importance_in_topk_community;
				}
			}
		}

		double sum = 0;
		for (int node = 0; node < nnodes; ++node) {
			tmp_pagerank[node] *= C;
			sum += tmp_pagerank[node];
		}
		// re-insert "leaked" pagerank
		double diff = 0.0, new_val;
		const double leaked = C - sum;
		for (int node = 0; node < nnodes; ++node) {
			if (node_info_for_topk[node].is_at_topk) {
				int comm = g.get_community(node);
				int topk_same_com = 0;
				topk_same_com = (comm == 0) ? (int)topk_blue.size() : (int)topk_red.size();
				new_val = tmp_pagerank[node] + ((leaked + 0.15) * (k / (double)nnodes) * g.get_community_percentage(comm) / (double)topk_same_com);
			} else {
				new_val = tmp_pagerank[node] + ((leaked + 0.15) / (double)nnodes);
			}
			diff += std::fabs(new_val - pagerankv[node].pagerank);
			pagerankv[node].pagerank = new_val;
		}
		if (diff < eps) break;
	}
	
	// Test print.
	double sum = 0; 
	double phi = 0;
	for (int i : topk_blue) sum += pagerankv[i].pagerank; 
	for (int i : topk_red) {sum += pagerankv[i].pagerank; phi += pagerankv[i].pagerank;}
	std::cout << "community 0 topk: " << 1 - phi / (double)sum << std::endl;
	std::cout << "community 1 topk: " << phi / (double)sum << std::endl;
	

	if (iter == max_iter) {
		std::cerr << "[WARN]: Local Fair Proporional Pagerank algorithm reached " << max_iter << " iterations." << std::endl;
	}
	return pagerankv;
}

// LFPRN Hybrid.
pagerank_v pagerank_algorithms::get_lfprh_topk(const int k, const double C, const bool use_cached, const double eps,
		const int max_iter)
{
	// Take pagerank so as to know the importance of each node. 
	pagerank_v pure_pagerank;
	if (use_cached && is_cache_valid)
		pure_pagerank = cached_pagerank;
	else
		pure_pagerank = get_pagerank(C, eps, max_iter);

	int nnodes = g.get_num_nodes();
	int ncommunities = g.get_num_communities();

	// Find the category that is unfavoured by pagerank. Works only for binary categories.
	double red_pagerank = g.get_pagerank_per_community(pure_pagerank)[1];
	const int unfavoured_category = (red_pagerank < g.get_community_percentage(1)) ? 1 : 0;
	// Sort pagerank vector so as to know the topk nodes.
	sort_pagerank_vector(pure_pagerank);
	// find top k of unfavoured kategory.
	std::vector<int> topk_unfavoured(k);
	int temp = 0;
	int node_category;
	// pure pagerank is sorted.
	for (pagerank_t &node : pure_pagerank) {
		node_category = g.get_community(node.node_id);

		if (node_category == unfavoured_category) {
			topk_unfavoured[k] = node.node_id;
			temp++;
		}
		if (temp == k) break;
	}

	// Calculate the custom excess policy/vector.
	

}

void pagerank_algorithms::set_personalization_type(personalization_t personalize_type, int extra_info)
{
	this->personalization_type = personalize_type;
	personalization_extra_info = extra_info;
	is_cache_valid = false;
}

void pagerank_algorithms::set_personalization_type(personalization_t personalize_type, std::vector<double> &jump_vector)
{
	this->personalization_type = personalize_type;
	this->jump_vector = jump_vector;
	is_cache_valid = false;
}

void pagerank_algorithms::compute_personalization_vector(std::vector<double> &pagerankv, double total_pagerank)
{
	switch (personalization_type)
	{
		case personalization_t::NO_PERSONALIZATION:
			compute_no_personalization_vector(pagerankv, total_pagerank);
			break;
		case personalization_t::ATTRIBUTE_PERSONALIZATION:
			compute_attribute_personalization_vector(pagerankv, total_pagerank, personalization_extra_info);
			break;
		case personalization_t::NODE_PERSONALIZATION:
			compute_node_personalization_vector(pagerankv, total_pagerank, personalization_extra_info);
			break;
		case personalization_t::JUMP_OPT_PERSONALIZATION:
			compute_custom_personalization_vector(pagerankv, total_pagerank);
			break;
		default:
			std::cerr << "Invalid personalization option." << std::endl;
			break;
	}
}

void pagerank_algorithms::compute_no_personalization_vector(std::vector<double> &pagerankv, double total_pagerank)
{
	#pragma omp parallel for
	for (unsigned int node = 0; node < pagerankv.size(); ++node) {
		const int comm = g.get_community(node);
		pagerankv[node] = total_pagerank * g.get_community_percentage(comm) / g.get_community_size(comm);
	}
}

void pagerank_algorithms::compute_pagerank_no_personalization_vector(std::vector<double> &pagerankv,
		double total_pagerank)
{
	const double new_pagerank = total_pagerank / (double)g.get_num_nodes();
	#pragma omp parallel for
	for (unsigned int node = 0; node < pagerankv.size(); ++node) {
		pagerankv[node] = new_pagerank;
	}
}

void pagerank_algorithms::compute_step_proportional_no_personalization_vector(std::vector<double> &pagerankv,
		double total_pagerank, std::vector<node_info_t> &node_info)
{
	#pragma omp parallel for
	for (unsigned int node = 0; node < pagerankv.size(); ++node) {
		const int comm = g.get_community(node);
		pagerankv[node] = total_pagerank * g.get_community_percentage(comm) * node_info[node].importance_in_community;;
	}
}

void pagerank_algorithms::compute_attribute_personalization_vector(std::vector<double> &pagerankv,
		double total_pagerank, int attribute_id)
{
	auto nodes = g.get_nodes_with_attribute(attribute_id);
	for (const auto &node_id : nodes) {
		pagerankv[node_id] = total_pagerank / (double)nodes.size();
	}
}

void pagerank_algorithms::compute_node_personalization_vector(std::vector<double> &pagerankv,
		double total_pagerank, int node_id)
{
	if ((node_id >= 0) && ((unsigned long)node_id < pagerankv.size()))
		pagerankv[node_id] = total_pagerank;
}

void pagerank_algorithms::compute_custom_personalization_vector(std::vector<double> &pagerankv,
		double total_pagerank)
{
	int nnodes = g.get_num_nodes();
	for (int i = 0; i <nnodes; i++) {
		pagerankv[i] = total_pagerank * jump_vector[i];
	}
}

void pagerank_algorithms::sort_pagerank_vector(pagerank_v &pagerank)
{
	std::sort(pagerank.begin(), pagerank.end());
}

void pagerank_algorithms::save_local_metrics(std::string out_filename_prefix, pagerank_v &pagerankv,
		bool orig_pg)
{
	std::ofstream outfile("out_" + out_filename_prefix + "_metrics.txt");
	//std::ofstream outfile0("out_" + out_filename_prefix + "_metrics_0.txt");
	//std::ofstream outfile1("out_" + out_filename_prefix + "_metrics_1.txt");
	//outfile0 << "fairn\tw_fairn" << std::endl;
	//outfile1 << "fairn\tw_fairn" << std::endl;
	outfile << std::fixed;
	outfile << std::setprecision(9);

	const int nnodes = g.get_num_nodes();
	const int ncommunities = g.get_num_communities();
	std::vector<double> in_homophily(nnodes);
	std::vector<double> out_homophily(nnodes);
	static std::unordered_map<int, int> orig_pagerank; // key: node_id, value: position in pagerank
	static std::unordered_map<int, double> map_pagerank; // key: node_id, value: pagerank

	if (orig_pg)
		for (int i = 0; i < nnodes; ++i)
			orig_pagerank.insert({pagerankv[i].node_id, i});

	for (int i = 0; i < nnodes; ++i)
		map_pagerank.insert({pagerankv[i].node_id, pagerankv[i].pagerank});

	// write file preamble
	outfile << "node_id\tpagerank";
	for (int comm = 0; comm < ncommunities; ++comm)
		outfile << "\texcess_" << comm;
	outfile << "\tin_homophily\tout_homophily";
	for (int comm = 0; comm < ncommunities; ++comm)
		outfile << "\tnneig_" << comm;
	for (int comm = 0; comm < ncommunities; ++comm)
		outfile << "\tneig_hom_" << comm;
	outfile << "\tmin_pos\tmed_pos\tmax_pos\tpos_diff\tfairn\tw_fairn" << std::endl;

	// calculate in and out homophily for each node
	for (int node = 0; node < nnodes; ++node) {
		const int my_comm = g.get_community(node);
		/* in_homophily = number of in-edges comming from the other
		 * communities divided by the number of in-edges */
		double curr_in_homophily = (g.get_in_degree(node) == 0) ? 0.5 :
			(g.get_in_degree(node) - g.count_in_neighbors_with_community(node, my_comm)) / (double)g.get_in_degree(node);
		in_homophily[node] = curr_in_homophily;
		double curr_out_homophily = (g.get_out_degree(node) == 0) ? 0.5 :
			(g.get_out_degree(node) - g.count_out_neighbors_with_community(node, my_comm)) / (double)g.get_out_degree(node);
		out_homophily[node] = curr_out_homophily;
	}

	// write rest of the file
	double total_fairout = 0.0, total_neigh_fairnout = 0.0;
	for (int i = 0; i < nnodes; ++i)
	{
		const int node = pagerankv[i].node_id;
		outfile << node << '\t' << pagerankv[i].pagerank;
		for (int comm = 0; comm < ncommunities; ++comm)
		{
			/* excess = percentage of all nodes with current community
			 * minus percentage of out neighbors with current community */
			double comm_percentage = g.get_community_size(comm) / (double)nnodes;
			double real_percentage = (g.get_out_degree(node) == 0) ? 0.0 :
				g.count_out_neighbors_with_community(node, comm) / (double)g.get_out_degree(node);
			double excess = comm_percentage - real_percentage;
			outfile << '\t' << excess;
		}
		outfile << '\t' << in_homophily[node] << '\t' << out_homophily[node];

		for (int comm = 0; comm < ncommunities; ++comm)
			outfile << '\t' << g.count_in_neighbors_with_community(node, comm);

		std::vector<double> sum_homophily(ncommunities);
		for (const auto &neigh : g.get_in_neighbors(node))
			sum_homophily[g.get_community(neigh)] += out_homophily[neigh];
		for (int comm = 0; comm < ncommunities; ++comm)
			if (g.count_in_neighbors_with_community(node, comm) == 0)
				outfile << '\t' << 0.0;
			else
				outfile << '\t' << sum_homophily[comm] / g.count_in_neighbors_with_community(node, comm);

		std::vector<int> neigh_positions;
		for (const auto &neigh : g.get_in_neighbors(node))
			neigh_positions.push_back(orig_pagerank[neigh]);
		std::sort(neigh_positions.begin(), neigh_positions.end());
		const int size = neigh_positions.size();
		if (size == 0)
			outfile << "\t-\t-\t-";
		else
		{
			int median = (size % 2 == 0) ? (neigh_positions[size/2 -1] + neigh_positions[size/2]) / 2.0 :
				neigh_positions[size/2];
			outfile << '\t' << neigh_positions[0] << '\t' << median << '\t' << neigh_positions[size-1];
		}
		outfile << '\t' << i - orig_pagerank[node];

		double sum_fairness = 0.0, sum_weighted_fairness = 0.0, sum_pagerank = 0.0;
		for (const auto &neigh : g.get_in_neighbors(node))
		{
			sum_fairness += 1 - out_homophily[neigh];
			sum_weighted_fairness += map_pagerank[neigh] * (1 - out_homophily[neigh]);
			sum_pagerank += map_pagerank[neigh];
		}
		double average_fairness = (g.get_in_degree(node) == 0) ? 0.5 :
			sum_fairness / g.get_in_degree(node);
		double average_weighted_fairness = (g.get_in_degree(node) == 0) ? 0.5 :
			sum_weighted_fairness / sum_pagerank;
		total_fairout += 1 - out_homophily[node];
		total_neigh_fairnout += average_fairness;
		outfile << '\t' << average_fairness << '\t' << average_weighted_fairness << std::endl;
	}

	outfile.close();
	//outfile0.close();
	//outfile1.close();
}
