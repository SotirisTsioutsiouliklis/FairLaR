/**
 * A simple stochastic algorithm for residual ptimization. Works good
 * enough when we start from a good initial point (proporional or
 * uniform - request prior knowledge.). Gives the options for initial
 * point: proportional, uniform, random. Great disadvandage the fact that
 * is based on the wrong theory that the solution domain is convex.
 * It should have some kind of restart.
 * 
 * TODO:
 *  i. Add restart. Solution's domain not convex.
 */
#include <fstream>
#include "residual_stochastic_opt.hpp"
#include <omp.h>
#include <chrono>
#include <iomanip>

// Parameters
int MAX_ITERATIONS = 300;
int NUMBER_OF_DIRECTIONS = 100;
double PRECISION_OF_SOLUTION = pow(10, -8);
double PRECISION_OF_CAT_RATIO = pow(10, -4);
int TIMES_TO_TRY_FOR_INITIAL_POINT = 100;
double INITIAL_STEP = 1.0;
int SMALLER_ALLOWED_STEP = pow(10, -10);

// Timing various intervals.
double loss_calc_time = 0;
std::chrono::time_point<std::chrono::high_resolution_clock> loss_start_time;
std::chrono::time_point<std::chrono::high_resolution_clock> loss_stop_time;
std::chrono::duration<double> elapsed_time;

// Save results.
void save_results(pagerank_v &fair_pagerank, std::vector<double> &jump_vector) {
    std::ofstream pagerank_file;
    pagerank_file.open("out_excess_sensitive_pagerank.txt");
    std::ofstream vector_file;
    vector_file.open("out_excess_sensitive_policy_v.txt");

    for (const auto &node : fair_pagerank) {
		pagerank_file << node.pagerank << std::endl;
	}

    std::vector<double>::iterator element;
    for (element = jump_vector.begin(); element < jump_vector.end(); element++) {
        vector_file << *element << std::endl;
    }
    pagerank_file.close();
    vector_file.close();
}

// Load Vector From File.
std::vector<double> load_point() {
    std::ifstream point ("out_excess_sensitive_policy_v.txt");
    std::vector<double> init_p;
    std::string line;
    while (getline(point ,line)) {
        init_p.push_back(std::stod(line));
    }

    return init_p;
}

// Euklidean norm.
double get_euklidean_norm(std::vector<double> &vec, int dimension) {
    double sum = 0;
    for (int i = 0; i < dimension; i++) {
        sum += pow(vec[i],2);
    }    

    return sqrt(sum);
}

// check if it is reduction direction.
bool is_reduction_direction(double current_loss, double temp_loss) {
    if (temp_loss < current_loss) {
        return true;
    } else {
        return false;
    }
}

// Check if result is fair.
bool is_fair_pagerank(pagerank_v &temp_pagerank, graph &g, double phi) {
    double red_ratio = g.get_pagerank_per_community(temp_pagerank)[1];
    if ((abs(red_ratio - phi) < PRECISION_OF_CAT_RATIO)) {
        return true;
    } else {
        return false;
    }
}

// Check if point is valid residual policy.
bool is_valid_residual_policy(std::vector<double> &point, graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double>::iterator counter;
    double red_sum = 0;
    double blue_sum = 0;

    for (int i = 0; i < dimension; i++) {
        // Sums each category's probabilities.
        (g.get_community(i) == 1) ? red_sum += point[i] : blue_sum += point[i];
        // Checks for coordinates to have the sense of probability.
        if (point[i] < 0 || point[i] > 1) {
            return false;
        }
    }

    // Checks for each category's probabilities to have the
    // sense of probability distribution.
    if (abs(red_sum - 1) > PRECISION_OF_CAT_RATIO || abs(blue_sum - 1) > PRECISION_OF_CAT_RATIO) {
        return false;
    }

    return true;
}

// Stores at <point> the candidate_point = current + step_direction.
void get_candidate_point(std::vector<double> &current_point, std::vector<double> &step_direction, std::vector<double> &point, int dimension) {
    for (int i = 0; i < dimension; i++) {
        point[i] = current_point[i] + step_direction[i];
    }
}

// Multiply vector by scalar. Returns step * direction.
std::vector<double> get_step_direction(double step, std::vector<double> &direction, int dimension) {
    std::vector<double> step_direction(dimension);
    for (int i = 0; i < dimension; i++) {
        step_direction[i] = step * direction[i];
    }

    return step_direction;
}

// Returns a jump vector. //Total = 2 --- Fix me.
std::vector<double> get_random_initial_point(graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double> initial_point(dimension, 0);
    double total_quantity = 1;
    double total_quantity_to_red = 1;
    double total_quantity_to_blue = 1;
    double quantity_to_give = 0;
    double max_quantity_to_give = 1;
    std::vector<int> nodes(dimension);
    for (int i = 0; i < dimension; i ++) {
        nodes[i] = i;
    }
    std::random_shuffle(nodes.begin(), nodes.end());

    int node = 0;
    while (total_quantity > pow(10, -4)) {
        if (g.get_community(nodes[node]) == 1) {
            max_quantity_to_give = std::min(total_quantity_to_red, 1 - initial_point[nodes[node]]);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_quantity_to_give;
            initial_point[nodes[node]] += quantity_to_give;
            total_quantity_to_red -= quantity_to_give;    
        } else {
            max_quantity_to_give = std::min(total_quantity_to_blue, 1 - initial_point[nodes[node]]);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_quantity_to_give;
            initial_point[nodes[node]] += quantity_to_give;
            total_quantity_to_blue -= quantity_to_give;
        }
        total_quantity -= quantity_to_give;
        node ++;
        if (node == dimension) {
            node = 0;
        } 
    }
    for (int i = 0; i < dimension; i++) {
        if ((1 - initial_point[nodes[i]] > total_quantity_to_red) && (g.get_community(nodes[i]) == 1)) {
            initial_point[nodes[i]] += total_quantity_to_red;
            break;
        }
    }
    for (int i = 0; i < dimension; i++) {
        if ((1 - initial_point[nodes[i]] > total_quantity_to_blue) && (g.get_community(nodes[i]) == 0)) {
            initial_point[nodes[i]] += total_quantity_to_blue;
            break;
        }
    }
    //std::cout << "Initailized\n";
    
    double red_sum = 0;
    double blue_sum = 0;
    for (int i = 0; i < dimension; i++) {
        if (g.get_community(i) == 1) {
            red_sum += initial_point[i];
        } else {
            blue_sum += initial_point[i];
        }
    }
    //std::cout << "Red ratio: " << red_sum << "blue ratio: " << blue_sum << "\n";
    

    return initial_point;
}

// Returns the unform excess policy.
std::vector<double> get_uniform_initial_point(graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double> initial_point(dimension, 0);
    for (int i = 0; i < g.get_num_nodes(); i++) {
        if (g.get_community(i) == 0) {
            initial_point[i] = 1 / (double)g.get_community_size(0);
        } else {
            initial_point[i] = 1 /(double)g.get_community_size(1);
        }
    }
    return initial_point;
}

// Returns valide direction.
std::vector<double> create_random_direction(std::vector<double> &current_point, graph &g) {
    int dimension = g.get_num_nodes();
    std::vector<double> direction(dimension, 0);
    std::vector<double> temp_point(dimension);
    std::vector<int> nodes(dimension);
    double quantity_to_add = 0;
    double quantity_to_add_red = 0;
    double quantity_to_add_blue = 0;
    double node_quantity = 0;
    double quantity_to_take = 0;
    double quantity_to_give = 0;
    double max_to_give;
    int node;

    for (int i = 0; i < dimension; i ++) {
        nodes[i] = i;
    }

    for (int i = 0; i < dimension; i++) {
        node_quantity = current_point[i];
        if (node_quantity > 0) {
            quantity_to_take = ((double)rand() / RAND_MAX) * node_quantity;
            direction[i] -= quantity_to_take;
            if (g.get_community(i) == 1) {
                quantity_to_add_red += quantity_to_take;
            } else {
                quantity_to_add_blue += quantity_to_take;
            }
            quantity_to_add += quantity_to_take;
        }
    }
  
    std::random_shuffle(nodes.begin(), nodes.end());
    get_candidate_point(current_point, direction, temp_point, dimension);

    int i = 0;
    while (quantity_to_add > pow(10, -4)) {
        node = nodes[i];
        if (g.get_community(node) == 1) {
            max_to_give = std::min((1-temp_point[i]), quantity_to_add_red);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_to_give;
            direction[node] += quantity_to_give;
            quantity_to_add_red -= quantity_to_give;
        } else {
            max_to_give = std::min((1-temp_point[i]), quantity_to_add_blue);
            quantity_to_give = ((double)rand() / RAND_MAX) * max_to_give;
            direction[node] += quantity_to_give;
            quantity_to_add_blue -= quantity_to_give;
        }
        quantity_to_add -= quantity_to_give;
        i++;
        if (i == dimension) {
            i = 0;
            get_candidate_point(current_point, direction, temp_point, dimension);
        }
    }

    std::random_shuffle(nodes.begin(), nodes.end());
    for (int i = 0; i < dimension; i ++) {
        node = nodes[i];
        if (1 - temp_point[node] > quantity_to_add) {
            if (g.get_community(i) == 1) {
                direction[node] += quantity_to_add_red;
                quantity_to_add_red = 0;
                quantity_to_add -= quantity_to_add_red;
            } else {
                direction[node] += quantity_to_add_blue;
                quantity_to_add_blue = 0;
                quantity_to_add -= quantity_to_add_blue;
            }
        }
    }
    get_candidate_point(current_point, direction, temp_point, dimension);
    double red_sum = 0;
    double blue_sum = 0;
    for (int i = 0; i < dimension; i++) {
        if (g.get_community(i) == 1) {
            red_sum += temp_point[i];
        } else {
            blue_sum += temp_point[i];
        }
    };

    return direction;
}

double find_max_step(pagerank_algorithms &algs , graph &g, std::vector<double> &current_point, std::vector<double> &direction,
     pagerank_v &pagerank, double &temp_loss_function_value, double &current_loss_function_value) 
{
    int dimension = g.get_num_nodes();
    std::vector<double> point(dimension);
    std::vector<double> step_direction(dimension);
    pagerank_v temp_pagerank;
    bool change_sign = true;

    // Find reduction direction.
    double step = INITIAL_STEP;
    // Return step * direction.
    step_direction = get_step_direction(step, direction, dimension);
    // Stores at <point> the candidate_point = current + step_direction.
    get_candidate_point(current_point, step_direction, point, dimension);
    // Start time for loss calculation.
    loss_start_time = std::chrono::high_resolution_clock::now();
    // Get temp custmo LFPR and its loss value at candidate_point.
    temp_pagerank = algs.get_custom_step_fair_pagerank(point);
    temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
    // Stop time for loss calculation.
    loss_stop_time = std::chrono::high_resolution_clock::now();
    // Renew duration for loss calculation.
    elapsed_time = loss_stop_time - loss_start_time;
    loss_calc_time += elapsed_time.count();

    // Find step (alogn with sign) for reduction direction.
    while(!is_reduction_direction(current_loss_function_value, temp_loss_function_value)) {
        if (change_sign) {
            step = - step;
        } else {
            step = step/(double)2;
        }
        change_sign = !change_sign;

        // Same as before.
        step_direction = get_step_direction(step, direction, dimension);
        get_candidate_point(current_point, step_direction, point, dimension);
        // Start time for loss calculation.
        loss_start_time = std::chrono::high_resolution_clock::now();
        // Get temp custmo LFPR and its loss value at candidate_point.
        temp_pagerank = algs.get_custom_step_fair_pagerank(point);
        temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
        // Stop time for loss calculation.
        loss_stop_time = std::chrono::high_resolution_clock::now();
        // Renew duration for loss calculation.
        elapsed_time = loss_stop_time - loss_start_time;
        loss_calc_time += elapsed_time.count();

        // If step is smaller than allowed, return 0 step
        // (i.e. stay at the same point).
        if (abs(step) < SMALLER_ALLOWED_STEP) {
            std::cout << "zero step";
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
            temp_pagerank = algs.get_custom_step_fair_pagerank(point);
            // Important to renew the temp_loss_function_value.
            // Not in this control, but keep it for cohesion.
            temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);

            return step;
        }
    }

    // Find valid residual policy.
    if (step < 0) {
        while (!is_valid_residual_policy(point, g) && abs(step) > SMALLER_ALLOWED_STEP) {
            step = step/(double)2;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
        }
        if (abs(step) < SMALLER_ALLOWED_STEP) {
            std::cout << "zero step";
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
        }
        temp_pagerank = algs.get_custom_step_fair_pagerank(point);
        temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
    }

    return step;
}

double loss_function_at(pagerank_v &pagerank, pagerank_v &temp_fair_pagerank, int dimension) {
    double sum = 0;
    for (int i = 0; i < dimension; i++) {
        sum += pow((pagerank[i].pagerank - temp_fair_pagerank[i].pagerank), 2);
    }

    return sum;
}

bool end_condition(double previous_value, double current_value) {
    // If we don't have significant change for 10
    // consecutive calls of function (iterations of algorithm)
    // return true (stop the program).
    static int no_important_change = 0;
    if (abs(previous_value - current_value) < PRECISION_OF_SOLUTION) {
        no_important_change++;
    } else {
        no_important_change = 0;
    }
    if (no_important_change == 10) {
        // For future implementation with restarts instead
        // of terminating.
        no_important_change = 0;
        return true;
    } else {
        return false;
    }
}

int main(int argc, char **argv) {
    // Define number of threads to use.
    omp_set_num_threads(20);

    // Initializations for time measures.
    //std::ofstream time_file;
    //std::ofstream converge_file;
    // For total program.
    //std::chrono::time_point<std::chrono::high_resolution_clock> total_start_time, total_stop_time;
    //double total_time = 0;
    // For Converge.
    //std::chrono::time_point<std::chrono::high_resolution_clock> con_start_time, con_stop_time;
    // For line search.
    //std::chrono::time_point<std::chrono::high_resolution_clock> line_start_time, line_stop_time;
    //double dir_srch_time = 0;
    // For direction search.
    //std::chrono::time_point<std::chrono::high_resolution_clock> dir_start_time, dir_stop_time;
    //double line_srch_time = 0;
    // Start total timer.
    //total_start_time = std::chrono::high_resolution_clock::now();

    // Read Command line arguments.
    double phi;
    if (argc == 1) {
		std::cout << "Provide <phi: wanted ratio for category 1>";
        return 0;
	} else if (argc == 2) {
        phi = std::atof(argv[1]);
    } else {
		std::cout << "Provide ONLY ONE ARGUMENT <phi: wanted ratio for category 1>";
        return 0;
    }
    
    // Initialize graph object
    graph g("out_graph.txt", "out_community.txt"); // Load graph.

    // Check phi.
    phi = (phi == 0) ? g.get_community_percentage(1) : phi;
    // Create phi file which is needed for phi != r.
    std::ofstream phi_file;
    phi_file.open("out_phi.txt");
    phi_file << "0\t" << 1 - phi << "\n";
    phi_file << "1\t" << phi;
    phi_file.close();
    // Load wanted ratio for categories.
    g.load_community_percentage("out_phi.txt");

    // Initializations.
    srand(time(NULL)); // Seed for rand() from clock.
    pagerank_algorithms algs(g); // Init algorithms class.
    int number_of_nodes = g.get_num_nodes(); // Dimension of points.
    pagerank_v temp_fair_pagerank; // For temp_best in each iteration.
    pagerank_v pagerank; // Pagerank vector.

    // Get pagerank vector.
    algs.set_personalization_type(personalization_t::NO_PERSONALIZATION, 0);
    pagerank = algs.get_pagerank();

    // For iterations.
    std::vector<double> current_point(number_of_nodes);
    std::vector<double> candidate_point(number_of_nodes);
    std::vector<double> candidate_direction(number_of_nodes);
    std::vector<double> temp_direction(number_of_nodes);
    pagerank_v current_pagerank;
    double current_loss_function_value;
    //double previous_loss_function_value; // For end condition that doesn't working properly. Fix me!!!
    double candidate_loss_value;
    double temp_loss_function_value;
    double temp_step;
    int whole_iterations = 0;

    // Initialize start point.
    current_point = algs.get_proportional_excess_vector();
    //current_point = get_uniform_initial_point();
    //current_point = get_random_initial_point();

    // Get custom LFPR.
    current_pagerank = algs.get_custom_step_fair_pagerank(current_point);
    current_loss_function_value = loss_function_at(pagerank, current_pagerank, number_of_nodes);
    candidate_loss_value = current_loss_function_value;
    //previous_loss_function_value = current_loss_function_value;

    // Write in converge file.
    //std::string name = std::string("out_stochastic_converge_") + std::to_string(NUMBER_OF_DIRECTIONS) + std::string(".txt");
    //converge_file.open(name);
    //converge_file << "iter\ttime\t\tloss" << std::endl;
    //converge_file << std::fixed;
    //converge_file << std::setprecision(9);
    //con_start_time = std::chrono::high_resolution_clock::now();
    //con_stop_time = con_start_time;
    //elapsed_time = con_stop_time - con_start_time;
    //converge_file << whole_iterations << "\t" << elapsed_time.count() << "\t" << current_loss_function_value << std::endl;

    // Start search iterations.
    while (whole_iterations < MAX_ITERATIONS) {
        // Print for knowing the stage for big Networks
        // that program is slow.
        std::cout << "Iteration:" << whole_iterations << "\n";
        whole_iterations++;
        
        // If no significance improvment in terms of solution's
        // presicion, stop.
        /*
        if (end_condition(previous_loss_function_value, current_loss_function_value)) {
            // Calculate total time.
            total_stop_time = std::chrono::high_resolution_clock::now();
            elapsed_time = total_stop_time - total_start_time;
            total_time = elapsed_time.count();
            // Write in time file.
            time_file.open("out_stochastic_timing.txt");
            time_file << std::fixed;
            time_file << std::setprecision(9);
            time_file << "total\t\tdir_srch\tloss_calc\tline_srch\n";
            time_file << total_time << "\t" << dir_srch_time << "\t" << loss_calc_time << "\t" << line_srch_time << std::endl;
            time_file.close();
            std::cout<< "-----------------end condition-----------------";
            exit(1);
        }
        */
        
        
        // Find the best direction and corresponding step.
        for (int i = 0; i < NUMBER_OF_DIRECTIONS; i++) {
            //std::cout << "New direction\n";
            // Start time for direction searching.
            //dir_start_time = std::chrono::high_resolution_clock::now();
            // Get random direction.
            temp_direction = create_random_direction(current_point, g);
            // Stop time for direction searching.
            //dir_stop_time = std::chrono::high_resolution_clock::now();
            // Renew duration for direction searching.
            //elapsed_time = dir_stop_time - dir_start_time;
            //dir_srch_time += elapsed_time.count();
            // Start time for line search.
            //line_start_time = std::chrono::high_resolution_clock::now();
            // Find feasible best step with bisection.
            // Renew temp_loss_function_value
            temp_step = find_max_step(algs, g, current_point, temp_direction, pagerank, temp_loss_function_value, current_loss_function_value);
            // Stop time for line search.
            //line_stop_time = std::chrono::high_resolution_clock::now();
            // Renew duration for line search.
            //elapsed_time = line_stop_time - line_start_time;
            //line_srch_time += elapsed_time.count();
            // Check if it is better than the current candidate point.
            if (temp_loss_function_value < candidate_loss_value) {
                // If it is.
                // Renew candidate loss.
                candidate_loss_value = temp_loss_function_value;
                candidate_direction = get_step_direction(temp_step, temp_direction, number_of_nodes);
                // Renew the candidate point.
                get_candidate_point(current_point, candidate_direction, candidate_point, number_of_nodes);
            }
        }

        // Renew values.
        current_point = candidate_point;
        //previous_loss_function_value = current_loss_function_value;
        current_loss_function_value = candidate_loss_value;
        //std::cout << "loss------------------------->" << current_loss_function_value << std::endl;

        // Renew time interval.
        //con_stop_time = std::chrono::high_resolution_clock::now();
        //elapsed_time = con_stop_time - con_start_time;

        // Write in converge file.
        //converge_file << whole_iterations << "\t" << elapsed_time.count() << "\t" << current_loss_function_value << std::endl;
    }
    // Close convergence file file.
    //converge_file.close();

    // Get best custom LFPR so far.
    temp_fair_pagerank = algs.get_custom_step_fair_pagerank(current_point);

    // Save results.
    save_results(temp_fair_pagerank, current_point);

    // Calculate total time.
    //total_stop_time = std::chrono::high_resolution_clock::now();
    //elapsed_time = total_stop_time - total_start_time;
    //total_time = elapsed_time.count();

    // Write in time file.
    //name = std::string("out_stochastic_timing_") + std::to_string(NUMBER_OF_DIRECTIONS) + std::string(".txt");
    //time_file.open(name);
    //time_file << "total\t\tdir_srch\tloss_calc\tline_srch\n";
    //time_file << std::fixed;
    //time_file << std::setprecision(9);
    //time_file << total_time << "\t" << dir_srch_time << "\t" << loss_calc_time << "\t" << line_srch_time << std::endl;
    //time_file.close();

    // Test print. You can ignore.
    //std::cout << "red ratio: " << g.get_pagerank_per_community(temp_fair_pagerank)[1];

    return 0;
}