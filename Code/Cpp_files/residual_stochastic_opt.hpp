#ifndef _residual_stochastic_opt_HPP
#define _residual_stochastic_opt_HPP

#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include "graph.hpp"
#include "pagerank.hpp"

double get_euklidean_norm(std::vector<double> &vec, int dimension);
std::vector<double> get_step_direction(double step, std::vector<double> &direction, int dimension);
std::vector<double> get_random_initial_point(graph &g);
std::vector<double> get_uniform_initial_point(graph &g);
std::vector<double> create_random_direction(std::vector<double> &current_point, graph &g);
double find_max_step(pagerank_algorithms &algs , graph &g, std::vector<double> &current_point, std::vector<double> &direction,
    pagerank_v &pagerank, double &temp_loss_function_value, double &current_loss_function_value);
double loss_function_at(pagerank_v &pagerank, pagerank_v &temp_fair_pagerank, int dimension);
bool restart_condition(double previous_value, double current_value);
bool is_probability_vector(std::vector<double> &point, graph &g);
void save_results(pagerank_v &fair_pagerank, std::vector<double> &jump_vector);
void get_candidate_point(std::vector<double> &current_point, std::vector<double> &step_direction, std::vector<double> &point, int dimension);
bool is_reduction_direction(double current_loss, double temp_loss);
bool is_fair_pagerank(pagerank_v &temp_pagerank, graph &g, double phi);

#endif