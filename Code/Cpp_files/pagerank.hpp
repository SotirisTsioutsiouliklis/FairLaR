#ifndef _PAGERANK_HPP
#define _PAGERANK_HPP

#include <vector>
#include "graph.hpp"

typedef struct {
	double node_percentage; // percentage of pagerank this node will GIVE to each of its neighbors
	std::vector<double> community_percentage; // percentage of pagerank this node wil GIVE to each community
	double importance_in_community; // percentage of community pagerank this node will GET
} node_info_t; // for step fair pagerank algorithms

enum class personalization_t {NO_PERSONALIZATION, ATTRIBUTE_PERSONALIZATION, NODE_PERSONALIZATION, JUMP_OPT_PERSONALIZATION};

class pagerank_algorithms
{
public:
	pagerank_algorithms(graph &g);

	pagerank_v get_pagerank(const double C=0.85, const double eps=1e-4, const int max_iter=100);
	pagerank_v get_global_uniform_fair_pagerank(const double C=0.85, const bool use_cached=true,
			const double eps=1e-4, const int max_iter=100);
	pagerank_v get_global_proportional_fair_pagerank(const double C=0.85,
			const bool use_cached=true, const double eps=1e-4, const int max_iter=100);
	pagerank_v get_step_uniform_fair_pagerank(const double C=0.85, const bool use_cached=true,
			const double eps=1e-4, const int max_iter=100);
	pagerank_v get_step_proportional_fair_pagerank(const double C=0.85, const bool use_cached=true,
			const double eps=1e-4, const int max_iter=100);
	pagerank_v get_custom_step_fair_pagerank(std::vector<double> &custom_excess, const double C=0.85, const bool use_cached=true,
			const double eps=1e-4, const int max_iter=100, const bool run_uniform=true);
	pagerank_v get_local_fair_pagerank(const double C=0.85, const double eps=1e-4,
			const int max_iter=100);
	// LFPR topk.
	pagerank_v get_lfprn_topk(const int k, const double C=0.85, const bool use_cached=true, const double eps=1e-4,
			const int max_iter=100);
	pagerank_v get_lfpru_topk(const int k, const double C=0.85, const bool use_cached=true, const double eps=1e-4,
			const int max_iter=100);
	pagerank_v get_lfprp_topk(const int k, const double C=0.85, const bool use_cached=true, const double eps=1e-4,
			const int max_iter=100);

	void set_personalization_type(personalization_t personalize_type, int extra_info);
	void set_personalization_type(personalization_t personalize_type, std::vector<double> &jump_vector);
	void compute_personalization_vector(std::vector<double> &pagerankv, double total_pagerank);

	void sort_pagerank_vector(pagerank_v &pagerank);
	void save_local_metrics(std::string out_filename_prefix, pagerank_v &pagerankv, bool orig_pg=false);
	std::vector<double> get_proportional_excess_vector(const double C=0.85, const bool use_cached=true,
			const double eps=1e-4, const int max_iter=100);

private:

	pagerank_v get_step_fair_pagerank(const double C=0.85, const bool use_cached=true,
			const double eps=1e-4, const int max_iter=100, const bool run_uniform=true);
	void initialize_node_info(std::vector<node_info_t> &node_info, const bool use_cached_pagerank,
			const double C, const double eps, const int max_iter);
	void initialize_node_info_topk(std::vector<node_topk> &node_info);
	void compute_no_personalization_vector(std::vector<double> &pagerankv, double total_pagerank);
	void compute_pagerank_no_personalization_vector(std::vector<double> &pagerankv, double total_pagerank);
	void compute_step_proportional_no_personalization_vector(std::vector<double> &pagerankv, double total_pagerank,
			std::vector<node_info_t> &node_info);
	void compute_attribute_personalization_vector(std::vector<double> &pagerankv, double total_pagerank, int attribute_id);
	void compute_node_personalization_vector(std::vector<double> &pagerankv, double total_pagerank, int node_id);
	void compute_custom_personalization_vector(std::vector<double> &pagerankv,
		double total_pagerank);

	graph &g;

	personalization_t personalization_type;
	int personalization_extra_info;
	std::vector<double> jump_vector;

	pagerank_v cached_pagerank; // to avoid recomputing it for the global fair algorithms
	bool is_cache_valid;
};

#endif /* _PAGERANK_HPP */
