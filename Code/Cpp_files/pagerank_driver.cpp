#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "graph.hpp"
#include "pagerank.hpp"

static bool get_options(const int argc, char ** const argv, double &jump_prob,
		std::string &personalize_filename, personalization_t &personalize_type,
		int &personalize_extra, std::string &comm_percentage_filename, bool &top_k, int &k)
{
	if (argc == 1)
	{
		personalize_type = personalization_t::NO_PERSONALIZATION;
		jump_prob = 0.15;
	}
	else if ((argc == 3) || (argc == 4))
	{
		if (!std::strcmp(argv[1], "-pn"))
		{
			jump_prob = 0.15;
			personalize_type = personalization_t::NODE_PERSONALIZATION;
			personalize_extra = std::atoi(argv[2]);
		}
		else if (!std::strcmp(argv[1], "-c"))
		{
			jump_prob = 0.15;
			personalize_type = personalization_t::NO_PERSONALIZATION;
			comm_percentage_filename = argv[2];
		}
		else if (!std::strcmp(argv[1], "-tk"))
		{
			top_k = true;
			k = std::atoi(argv[2]);
			personalize_type = personalization_t::NO_PERSONALIZATION;
			jump_prob = 0.15;
		}
		else
			goto error;
	}
	else if (argc == 5)
	{
		if (!std::strcmp(argv[1], "-tk") && !std::strcmp(argv[3], "-c")) {
			top_k = true;
			k = std::atoi(argv[2]);
			jump_prob = 0.15;
			personalize_type = personalization_t::NO_PERSONALIZATION;
			comm_percentage_filename = argv[4];
		} else if (!std::strcmp(argv[1], "-pn") && !std::strcmp(argv[3], "-c")) {
			jump_prob = 0.15;
			personalize_type = personalization_t::NODE_PERSONALIZATION;
			personalize_extra = std::atoi(argv[2]);
			comm_percentage_filename = argv[4];
		} else {
			goto error;
		}
	}
	else
	{
		goto error;
	}
	return true;

error:
	std::cerr << "Usage: " << argv[0] << " [options]\n"
		"Options:\n"
		"-pn <node_id>\t\t\tnode id for node personalization\n"
		"-c <comm_percent_filename> \tfilename for custom community percentage\n"
		"-tk <K for fairness in top k> \tprovide integer > 0, < number of nodes" << std::endl;
	return false;
}

static std::string get_local_header(const bool is_local)
{
	if (is_local)
		return "\t|\tin_n0\t\%in_n0\tout_n0\t\%out_n0\tin_n1\t\%in_n1\tout_n1\t\%out_n1\t"
			"in_neighbors_of_community_i\t\%in_neighbors_of_community_i\t"
			"out_neighbors_of_community_i\t\%out_neighbors_of_community_i...";
	else
		return "";
}

static std::string get_local_neighbors(const bool is_local, const int node_id, graph &g)
{
	if (!is_local)
		return "";

	std::string result = "\t\t|";
	const int ncommunities = g.get_num_communities();
	const int n_in_neighbors = g.get_in_degree(node_id);
	const int n_out_neighbors = g.get_out_degree(node_id);

	for (int community = 0; community < ncommunities; ++community)
	{
		int nsame_in_neighbors = g.count_in_neighbors_with_community(node_id, community);
		int in_percentage = (n_in_neighbors == 0) ? 0 : nsame_in_neighbors / (double)n_in_neighbors * 100;
		int nsame_out_neighbors = g.count_out_neighbors_with_community(node_id, community);
		int out_percentage = (n_out_neighbors == 0) ? 0 : nsame_out_neighbors / (double)n_out_neighbors * 100;
		result.append("\t");
		result.append(std::to_string(nsame_in_neighbors));
		result.append("\t(");
		result.append(std::to_string(in_percentage));
		result.append("\%)\t");
		result.append(std::to_string(nsame_out_neighbors));
		result.append("\t(");
		result.append(std::to_string(out_percentage));
		result.append("\%)");
	}

	return result;
}

static void save_pagerank(std::string filename_prefix, pagerank_v &pagerankv, graph &g,
		pagerank_algorithms &algs, std::ofstream &out_summary)
{
	bool is_local = (filename_prefix == "local");
	bool is_pagerank = (filename_prefix == "pagerank");
	std::ofstream outfile_rank;
	outfile_rank.open("out_" + filename_prefix + "_rank.txt");
	//std::ofstream outfile_pro_pos;
	//outfile_pro_pos.open("out_" + filename_prefix + "_pos.txt");
	std::ofstream outfile_pagerank;
	outfile_pagerank.open("out_" + filename_prefix + "_pagerank.txt");
	//std::ofstream outfile_label_ranking;
	////outfile_label_ranking.open("out_" + filename_prefix + "_label_ranking.txt");
	//std::ofstream outfile_sums;
	//outfile_sums.open("out_" + filename_prefix + "_sums.txt");
	std::ofstream outfile_sums_perc;
	outfile_sums_perc.open("out_" + filename_prefix + "_sums_prec.txt");

	for (const auto &node : pagerankv) {
		outfile_pagerank << node.pagerank << std::endl;
	}

	algs.sort_pagerank_vector(pagerankv);

	//for (int node = 0; node < g.get_num_nodes(); ++node)
		//if (g.get_community(node) == 1)
			//outfile_pro_pos << node << std::endl;

	outfile_rank << std::fixed;
	outfile_rank << std::setprecision(9);
	outfile_rank << "# node\tpagerank\tcommunity" << get_local_header(is_local) << std::endl;

	//outfile_sums << std::fixed;
	//outfile_sums << std::setprecision(9);
	//outfile_sums << "# range\t\tcommunity_0\tcommunity_1\tcommunity_i..." << std::endl;
	outfile_sums_perc << std::fixed;
	outfile_sums_perc << std::setprecision(9);
	outfile_sums_perc << "# range\t\tcommunity_0\tcommunity_1\tcommunity_i..." << std::endl;

	std::vector<double> pagerank_sum(g.get_num_communities());
	double sum_pagerank = 0.0; // sum of pagerank for all communities so far
	int i;
	for (i = 0; i < g.get_num_nodes(); ++i)
	{
		const auto &node = pagerankv[i];
		pagerank_sum[g.get_community(node.node_id)] += node.pagerank;
		sum_pagerank += pagerankv[i].pagerank;
		outfile_rank << node.node_id << "\t" << node.pagerank << "\t" << g.get_community(node.node_id)
			<< get_local_neighbors(is_local, node.node_id, g) << std::endl;
		//outfile_label_ranking << g.get_community(node.node_id) << std::endl;
		if ((i > 0) && ((i + 1) % 10 == 0))
		{
			//outfile_sums << "1-" << i + 1 << "\t";
			outfile_sums_perc << "1-" << i + 1 << "\t";
			for (int community = 0; community < g.get_num_communities(); ++community)
			{
				//outfile_sums << "\t" << pagerank_sum[community];
				outfile_sums_perc << "\t" << pagerank_sum[community] / sum_pagerank;
			}
			//outfile_sums << std::endl;
			outfile_sums_perc << std::endl;
		}
	}
	if (g.get_num_nodes() % 10 != 0)
	{
		//outfile_sums << "1-" << i;
		outfile_sums_perc << "1-" << i;
		for (int community = 0; community < g.get_num_communities(); ++community)
		{
			//outfile_sums << "\t" << pagerank_sum[community];
			outfile_sums_perc << "\t" << pagerank_sum[community] / sum_pagerank;
		}
		//outfile_sums << std::endl;
		outfile_sums_perc << std::endl;
	}

	for (int community = 0; community < g.get_num_communities(); ++community)
	{
		std::cout << "Community: " << community << ", Pagerank: " << pagerank_sum[community] << std::endl;
		out_summary << "Community: " << community << ", Pagerank: " << pagerank_sum[community] << std::endl;
	}

	outfile_rank.close();
	//outfile_pro_pos.close();
	outfile_pagerank.close();
	//outfile_label_ranking.close();
	//outfile_sums.close();
	outfile_sums_perc.close();

	algs.save_local_metrics(filename_prefix, pagerankv, is_pagerank);
}

static void print_preamble(std::ofstream &out_summary, graph &g)
{
	std::cout << "Number of nodes: " << g.get_num_nodes() << std::endl;
	out_summary << "Number of nodes: " << g.get_num_nodes() << std::endl;
	std::cout << "Number of edges: " << g.get_num_edges() << "\n" << std::endl;
	out_summary << "Number of edges: " << g.get_num_edges() << "\n" << std::endl;

	for (int community = 0; community < g.get_num_communities(); ++community) {
		std::cout << "Community " << community << ": " << g.get_community_size(community) <<
			" nodes (" << g.get_community_size(community) / (double)g.get_num_nodes() * 100 << "%)" << std::endl;
		out_summary << "Community " << community << ": " << g.get_community_size(community) <<
			" nodes (" << g.get_community_size(community) / (double)g.get_num_nodes() * 100 << "%)" << std::endl;
	}
}

static void print_algo_info(std::string algo_name, std::ofstream &out_summary,
		personalization_t &personalize_type, double jump_prob, int extra_info)
{
	std::string personalize_type_s = "";
	if (personalize_type == personalization_t::ATTRIBUTE_PERSONALIZATION)
	{
		personalize_type_s = "attribute personalization (";
		personalize_type_s += std::to_string(extra_info);
		personalize_type_s += ") ";
	}
	else if (personalize_type == personalization_t::NODE_PERSONALIZATION)
	{
		personalize_type_s = "node personalization (";
		personalize_type_s += std::to_string(extra_info);
		personalize_type_s += ") ";
	}

	std::cout << "\nRunning " << personalize_type_s << algo_name <<
		" with jump probability = " << jump_prob << " ..." << std::endl;
	out_summary << "\nRunning " << personalize_type_s << algo_name <<
		" with jump probability = " << jump_prob << " ..." << std::endl;
}

int main(int argc, char **argv)
{
	double jump_prob;
	std::string personalize_filename, comm_percentage_filename = "";
	personalization_t personalize_type;
	int personalize_extra = 0;
	bool topk = false; // If -tk is provided.
	int k = 0; // The K for topk.
	if (!get_options(argc, argv, jump_prob, personalize_filename, personalize_type,
				personalize_extra, comm_percentage_filename, topk, k))
		return 1;

	std::ofstream out_summary("summary.txt");
	graph g("out_graph.txt", "out_community.txt");
	//Sotiris Tsiou Separate topk alforithms from total.
	if (topk)
	{
		if (comm_percentage_filename != "")
			g.load_community_percentage(comm_percentage_filename);

		pagerank_algorithms algs(g);
		algs.set_personalization_type(personalize_type, personalize_extra);

		print_preamble(out_summary, g);

		//Call pure pagerank.
		print_algo_info("Pagerank", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerank_v pagerankv = algs.get_pagerank(1 - jump_prob);
		save_pagerank("pagerank", pagerankv, g, algs, out_summary);

		print_algo_info("lfpr neighborhood topk", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_lfprn_topk(k, 1 - jump_prob);
		save_pagerank("lfpr_n_topk", pagerankv, g, algs, out_summary);

		print_algo_info("lfpr uniform topk", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_lfpru_topk(k, 1 - jump_prob);
		save_pagerank("lfpr_u_topk", pagerankv, g, algs, out_summary);

		print_algo_info("lfpr proportional topk", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_lfprp_topk(k, 1 - jump_prob);
		save_pagerank("lfpr_p_topk", pagerankv, g, algs, out_summary);

		print_algo_info("lfpru hybrid topk", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_lfprhu_topk(k, 1 - jump_prob);
		save_pagerank("lfpr_hu_topk", pagerankv, g, algs, out_summary);

		print_algo_info("lfprn hybrid topk", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_lfprhn_topk(k, 1 - jump_prob);
		save_pagerank("lfpr_hn_topk", pagerankv, g, algs, out_summary);
	} else 
	{
		if (personalize_type == personalization_t::ATTRIBUTE_PERSONALIZATION)
			g.load_attributes(personalize_filename);
		if (comm_percentage_filename != "")
			g.load_community_percentage(comm_percentage_filename);
		pagerank_algorithms algs(g);
		algs.set_personalization_type(personalize_type, personalize_extra);

		print_preamble(out_summary, g);

		print_algo_info("Pagerank", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerank_v pagerankv = algs.get_pagerank(1 - jump_prob);
		save_pagerank("pagerank", pagerankv, g, algs, out_summary);

		print_algo_info("global uniform Pagerank", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_global_uniform_fair_pagerank(1 - jump_prob);
		save_pagerank("global_uniform", pagerankv, g, algs, out_summary);

		print_algo_info("global proportional Pagerank", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_global_proportional_fair_pagerank(1 - jump_prob);
		save_pagerank("global_proportional", pagerankv, g, algs, out_summary);

		print_algo_info("lfpr uniform", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_step_uniform_fair_pagerank(1 - jump_prob);
		save_pagerank("lfpr_u", pagerankv, g, algs, out_summary);

		print_algo_info("lfpr proportional", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_step_proportional_fair_pagerank(1 - jump_prob);
		save_pagerank("lfpr_p", pagerankv, g, algs, out_summary);

		print_algo_info("local neighborhood", out_summary, personalize_type, jump_prob, personalize_extra);
		pagerankv = algs.get_local_fair_pagerank(1 - jump_prob);
		save_pagerank("lfpr_n", pagerankv, g, algs, out_summary);
	}
}
