/** 
 * Just debug purposes. Delet after.
*/
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "graph.hpp"
#include "pagerank.hpp"

static void save_pagerank(std::string filename_prefix, pagerank_v &pagerankv)
{
	std::ofstream outfile_pagerank;
	outfile_pagerank.open("out_" + filename_prefix + "_pagerank.txt");

	for (const auto &node : pagerankv) {
		outfile_pagerank << node.pagerank << std::endl;
	}

	outfile_pagerank.close();
}


int main() {
    pagerank_v pagerankv;
    graph g("out_graph.txt", "out_community.txt");
    pagerank_algorithms algs(g);
    algs.set_personalization_type(personalization_t::NO_PERSONALIZATION, 0);
    
    pagerankv = algs.get_step_proportional_fair_pagerank();
	save_pagerank("lfpr_p", pagerankv);

    std::vector<double> customExcess = algs.get_proportional_excess_vector();

    std::ofstream out_file;
    out_file.open("prop_excess.txt");

    for (double i : customExcess) {
        out_file << i << std::endl;
    }

    out_file.close();


    pagerankv = algs.get_custom_step_fair_pagerank(customExcess);
    save_pagerank("custom_prop", pagerankv);
}