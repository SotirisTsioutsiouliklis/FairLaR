#include <fstream>
#include "local_jump_optimization.hpp"

// Parameters
int FUNCTIONAL_CALCULATIONS = 10000;
int NUMBER_OF_DIRECTIONS = 100;
double PRECISION_OF_SOLUTION = pow(10, -8);
double PRECISION_OF_RED_RATIO = pow(10, -4);
int TIMES_TO_TRY_FOR_INITIAL_POINT = 100;
int iter = 0;
std::ofstream log_file;

// Save results.
void save_results(pagerank_v &fair_pagerank, std::vector<double> &jump_vector) {
    std::ofstream pagerank_file;
    pagerank_file.open("opt_mine_local_fairpagerank_vector.txt");
    std::ofstream vector_file;
    vector_file.open("opt_mine_local_jump_vector.txt");

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
    if ((abs(red_ratio - phi) < PRECISION_OF_RED_RATIO)) {
        return true;
    } else {
        return false;
    }
}

// Check if point is probability vector.
bool is_probability_vector(std::vector<double> &point) {
    std::vector<double>::iterator counter;
    double sum = 0;
    for (counter = point.begin(); counter < point.end(); counter++) {
        sum += *counter;
        if (*counter < 0 || *counter > 1) {
            return false;
        }
    }
    if (abs(sum - 1) > pow(10, -4)) {
        return false;
    }
    return true;
}

// returns current_point + step_direction
void get_candidate_point(std::vector<double> &current_point, std::vector<double> &step_direction, std::vector<double> &point, int dimension) {
    for (int i = 0; i < dimension; i++) {
        point[i] = current_point[i] + step_direction[i];
    }
}

// Multiply vector by scalar.
std::vector<double> get_step_direction(double step, std::vector<double> &direction, int dimension) {
    std::vector<double> step_direction(dimension);
    for (int i = 0; i < dimension; i++) {
        step_direction[i] = step * direction[i];
    }

    return step_direction;
}

// Returns a jump vector.
std::vector<double> get_random_initial_point(graph &g, double phi) {
    int dimension = g.get_num_nodes();
    std::vector<double> initial_point(dimension, 0);
    double total_quantity = 1;
    double total_quantity_to_red = phi;
    double total_quantity_to_blue = 1 - phi;
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
    std::cout << "Initailized\n";
    double red_sum = 0;
    double blue_sum = 0;
    for (int i = 0; i < dimension; i++) {
        if (g.get_community(i) == 1) {
            red_sum += initial_point[i];
        } else {
            blue_sum += initial_point[i];
        }
    }
    std::cout << "Red ratio: " << red_sum << "blue ratio: " << blue_sum << "\n";

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
            } else {
                direction[node] += quantity_to_add_blue;
            }
            break;
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
    }
    std::cout << "Candidate point:\n";
    std::cout << "Red ratio: " << red_sum << "blue ratio: " << blue_sum << "\n";

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
    double step = 1;
    step_direction = get_step_direction(step, direction, dimension);
    get_candidate_point(current_point, step_direction, point, dimension);
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, point);
    temp_pagerank = algs.get_local_fair_pagerank(); 
    iter++;
    temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
    while(!is_reduction_direction(current_loss_function_value, temp_loss_function_value)) {
        if (change_sign) {
            step = - step;
            change_sign = !change_sign;
        } else {
            step = step/(double)2;
            change_sign = !change_sign;
        }
        if (abs(step) < pow(10, -10)) {
            std::cout << "zero step";
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
            temp_pagerank = algs.get_custom_step_fair_pagerank(point);
            temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
            
            return step;
        }
        step_direction = get_step_direction(step, direction, dimension);
        get_candidate_point(current_point, step_direction, point, dimension);
        algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, point);
        temp_pagerank = algs.get_local_fair_pagerank(); 
        iter++;
        temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
    }
    // Find probability vector.
    if (step < 0) {
        while (!is_probability_vector(point)) {
            step = step/(double)2;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
        }
        if (abs(step) < pow(10, -10)) {
            std::cout << "zero step";
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
        }
        algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, point);
        temp_pagerank = algs.get_local_fair_pagerank();
    }
   
    temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);

    return step;
}

double loss_function_at(pagerank_v &pagerank, pagerank_v &temp_fair_pagerank, int dimension) {
    double sum = 0;
    for (int i = 0; i < dimension; i++) {
        sum += pow((pagerank[i].pagerank - temp_fair_pagerank[i].pagerank), 2);
    }

    return sum;
}

bool restart_condition(double previous_value, double current_value) {
    static int no_important_change = 0;
    if (abs(previous_value - current_value) < PRECISION_OF_SOLUTION) {
        no_important_change++;
    } else {
        no_important_change = 0;
    }
    if (no_important_change == 10) {
        no_important_change = 0;
        return true;
    } else {
        return false;
    }
}

int main(int argc, char **argv) {
    // Read command line arguments.
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

    log_file.open("log_file.txt");
    log_file << "Start experiments with phi: " << phi << "\n";
    // Initializations.
    srand(time(NULL)); // Init randomly
    ////std::cout << "read graph\n";
    graph g("out_graph.txt", "out_community.txt"); // Load graph.
    pagerank_algorithms algs(g); // Init algoriths class.
    int number_of_nodes = g.get_num_nodes(); // Dimension of points.
    std::vector<double> best_point(number_of_nodes, 0); // minimizer.
    double best_loss_value = 10; // Loss value of minimizer.
    pagerank_v temp_fair_pagerank;
    algs.set_personalization_type(personalization_t::NO_PERSONALIZATION, 0); // Pagerank.
    pagerank_v pagerank = algs.get_local_fair_pagerank();
    iter++;
    // For iterations.
    std::vector<double> current_point(number_of_nodes);
    std::vector<double> candidate_point(number_of_nodes);
    std::vector<double> candidate_direction(number_of_nodes);
    std::vector<double> temp_direction(number_of_nodes);
    pagerank_v current_pagerank;
    double current_loss_function_value;
    double candidate_loss_value;
    double temp_loss_function_value;
    double temp_step;
    int whole_iterations = 0;
restart:
    ////std::cout << "get initial point\n";
    current_point = get_random_initial_point(g, phi);
    log_file << "Initial point: [";
    for (int l = 0; l < number_of_nodes; l++) {
        log_file  << " " << current_point[l] << ",";
    }
    log_file << "\n";
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, current_point);
    current_pagerank = algs.get_local_fair_pagerank();
    iter++;
    current_loss_function_value = loss_function_at(pagerank, current_pagerank, number_of_nodes);
    candidate_loss_value = current_loss_function_value;

    if (current_loss_function_value < best_loss_value) {
        best_point = current_point;
        best_loss_value = current_loss_function_value;
    }
    
    log_file << "current loss value: " << current_loss_function_value << "\n";
    log_file << "best loss value: " << best_loss_value << "\n";  
    log_file << "Start iterations:\n";
    while (iter < FUNCTIONAL_CALCULATIONS) {
        whole_iterations++;
        std::cout << "New point\n";
        log_file << "Iteration: " << whole_iterations << "\n";
        
        if (restart_condition(best_loss_value, current_loss_function_value)) {
            log_file << "-------------------------------------------RESTART----------------------------------------\n";
            if (current_loss_function_value < best_loss_value) {
            best_point = current_point;
            best_loss_value = current_loss_function_value;
            }
            log_file << "current loss value: " << current_loss_function_value << "\n";
            log_file << "best loss value: " << best_loss_value << "\n";
            goto restart;
        }
        
        // Find the best direction and corresponding step.
        for (int i = 0; i < NUMBER_OF_DIRECTIONS; i++) {
            std::cout << "New direction\n";
            temp_direction = create_random_direction(current_point, g);
            temp_step = find_max_step(algs, g, current_point, temp_direction, pagerank, temp_loss_function_value, current_loss_function_value);
            // Check if it is better than the candidate point.
            if (temp_loss_function_value < candidate_loss_value) {
                candidate_loss_value = temp_loss_function_value;
                candidate_direction = get_step_direction(temp_step, temp_direction,number_of_nodes);
                get_candidate_point(current_point, candidate_direction, candidate_point, number_of_nodes);
            }
        }
        // Renew values.
        log_file << "End of direction searching.\nI was in point: [";
        log_file << "\n";
        log_file << "With loss value: " << current_loss_function_value << "\n";
        current_point = candidate_point;
        current_loss_function_value = candidate_loss_value;
        log_file << "I will go to point: [";
        log_file << "\n";
        log_file << "With loss value: " << current_loss_function_value << "\n";
        log_file << "Continue searching to improve new current point if i have functional calculations left\n"; 
    }
    log_file << "End of functional calculations - end of Searching.\n";
    // Renew values.
    if (current_loss_function_value < best_loss_value) {
        best_point = current_point;
        best_loss_value = current_loss_function_value;
    }

    log_file << "current loss value: " << current_loss_function_value << "\n";
    log_file << "best loss value: " << best_loss_value << "\n";
    log_file.close();
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, best_point);
    temp_fair_pagerank = algs.get_local_fair_pagerank();
    save_results(temp_fair_pagerank, best_point);
    std::cout << "red ratio: " << g.get_pagerank_per_community(temp_fair_pagerank)[1];

    return 0;
}

//--------------------------------------------------------------------------------------------

#include <fstream>
#include "local_optimization.hpp"

// Parameters
int FUNCTIONAL_CALCULATIONS = 5000;
int NUMBER_OF_DIRECTIONS = 20;
double PRECISION_OF_SOLUTION = pow(10, -8);
double PRECISION_OF_RED_RATIO = pow(10, -4);
int TIMES_TO_TRY_FOR_INITIAL_POINT = 100;
int iter = 0;
std::ofstream local_log_file;

// Save results.
void save_results(pagerank_v &fair_pagerank, std::vector<double> &jump_vector) {
    std::ofstream pagerank_file;
    pagerank_file.open("opt_mine_local_fairpagerank_vector.txt");
    std::ofstream vector_file;
    vector_file.open("opt_mine_local_jump_vector.txt");

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
    if ((abs(red_ratio - phi) < PRECISION_OF_RED_RATIO)) {
        return true;
    } else {
        return false;
    }
}

// Check if point is probability vector.
bool is_probability_vector(std::vector<double> &point) {
    std::vector<double>::iterator counter;
    double sum = 0;
    for (counter = point.begin(); counter < point.end(); counter++) {
        sum += *counter;
        if (*counter < 0 || *counter > 1) {
            return false;
        }
    }
    if (abs(sum - 1) > pow(10, -4)) {
        return false;
    }
    return true;
}

// returns current_point + step_direction
void get_candidate_point(std::vector<double> &current_point, std::vector<double> &step_direction, std::vector<double> &point, int dimension) {
    for (int i = 0; i < dimension; i++) {
        point[i] = current_point[i] + step_direction[i];
    }
}

// Multiply vector by scalar.
std::vector<double> get_step_direction(double step, std::vector<double> &direction, int dimension) {
    std::vector<double> step_direction(dimension);
    for (int i = 0; i < dimension; i++) {
        step_direction[i] = step * direction[i];
    }

    return step_direction;
}

// Returns a jump vector.
std::vector<double> get_random_initial_point(graph &g, double phi) {
    int dimension = g.get_num_nodes();
    std::vector<double> initial_point(dimension, 0);
    double total_quantity = 1;
    double total_quantity_to_red = phi;
    double total_quantity_to_blue = 1 - phi;
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
    /*
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
    */

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
            } else {
                direction[node] += quantity_to_add_blue;
                quantity_to_add_blue = 0;
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
    }
    //std::cout << "Candidate point:\n";
    //std::cout << "Red ratio: " << red_sum << "blue ratio: " << blue_sum << "\n";

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
    double step = 1;
    step_direction = get_step_direction(step, direction, dimension);
    get_candidate_point(current_point, step_direction, point, dimension);
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, point);
    temp_pagerank = algs.get_local_fair_pagerank(); 
    iter++;
    temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
    while(!is_reduction_direction(current_loss_function_value, temp_loss_function_value) && abs(step) > pow(10, -10)) {
        if (change_sign) {
            step = - step;
            change_sign = !change_sign;
        } else {
            step = step/(double)2;
            change_sign = !change_sign;
        }
        step_direction = get_step_direction(step, direction, dimension);
        get_candidate_point(current_point, step_direction, point, dimension);
        algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, point);
        temp_pagerank = algs.get_local_fair_pagerank(); 
        iter++;
        temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
    }
    
    if (abs(step) < pow(10, -10)) {
        step = 0;
        step_direction = get_step_direction(step, direction, dimension);
        get_candidate_point(current_point, step_direction, point, dimension);
    }
    // Find probability vector.
    if (step < 0) {
        while (!is_probability_vector(point) && abs(step) > pow(10, -10)) {
            step = step/(double)2;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
        }
        if (abs(step) < pow(10, -10)) {
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
        }
        algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, point);
        temp_pagerank = algs.get_local_fair_pagerank();
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

bool restart_condition(double previous_value, double current_value) {
    static int no_important_change = 0;
    if (abs(previous_value - current_value) < PRECISION_OF_SOLUTION) {
        no_important_change++;
    } else {
        no_important_change = 0;
    }
    if (no_important_change == 10) {
        no_important_change = 0;
        return true;
    } else {
        return false;
    }
}

int main(int argc, char **argv) {
    // Read command line arguments.
    double phi;
    if (argc == 1) {
		//std::cout << "Provide <phi: wanted ratio for category 1>";
        return 0;
	} else if (argc == 2) {
        phi = std::atof(argv[1]);
    } else {
		//std::cout << "Provide ONLY ONE ARGUMENT <phi: wanted ratio for category 1>";
        return 0;
    }

    local_log_file.open("local_log_file.txt");
    local_log_file << "Start experiments with phi: " << phi << "\n";
    // Initializations.
    srand(time(NULL)); // Init randomly
    //////std::cout << "read graph\n";
    graph g("out_graph.txt", "out_community.txt"); // Load graph.
    g.load_community_percentage("out_phi.txt");
    pagerank_algorithms algs(g); // Init algoriths class.
    int number_of_nodes = g.get_num_nodes(); // Dimension of points.
    std::vector<double> best_point(number_of_nodes, 0); // minimizer.
    double best_loss_value = 10; // Loss value of minimizer.
    pagerank_v temp_fair_pagerank;
    algs.set_personalization_type(personalization_t::NO_PERSONALIZATION, 0); // Pagerank.
    pagerank_v pagerank = algs.get_pagerank();
    iter++;
    // For iterations.
    std::vector<double> current_point(number_of_nodes);
    std::vector<double> candidate_point(number_of_nodes);
    std::vector<double> candidate_direction(number_of_nodes);
    std::vector<double> temp_direction(number_of_nodes);
    pagerank_v current_pagerank;
    double current_loss_function_value;
    double candidate_loss_value;
    double temp_loss_function_value;
    double temp_step;
    int whole_iterations = 0;
restart:
    //////std::cout << "get initial point\n";
    current_point = get_random_initial_point(g, phi);
    local_log_file << "Initial point: [";
    for (int l = 0; l < number_of_nodes; l++) {
        if (current_point[l] != 0) {
            local_log_file  << l << " : " << current_point[l] << "\n";
        }
    }
    local_log_file << "\n";
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, current_point);
    current_pagerank = algs.get_local_fair_pagerank();
    iter++;
    current_loss_function_value = loss_function_at(pagerank, current_pagerank, number_of_nodes);
    candidate_loss_value = current_loss_function_value;

    if (current_loss_function_value < best_loss_value) {
        best_point = current_point;
        best_loss_value = current_loss_function_value;
    }
    
    local_log_file << "current loss value: " << current_loss_function_value << "\n";
    local_log_file << "best loss value: " << best_loss_value << "\n";  
    local_log_file << "Start iterations:\n";
    while (iter < FUNCTIONAL_CALCULATIONS) {
        whole_iterations++;
        std::cout << "New point\n";
        local_log_file << "Iteration: " << whole_iterations << "\n";
        
        if (restart_condition(best_loss_value, current_loss_function_value)) {
            local_log_file << "-------------------------------------------RESTART----------------------------------------\n";
            if (current_loss_function_value < best_loss_value) {
            best_point = current_point;
            best_loss_value = current_loss_function_value;
            }
            local_log_file << "current loss value: " << current_loss_function_value << "\n";
            local_log_file << "best loss value: " << best_loss_value << "\n";
            goto restart;
        }
        
        // Find the best direction and corresponding step.
        for (int i = 0; i < NUMBER_OF_DIRECTIONS; i++) {
            //std::cout << "New direction\n";
            temp_direction = create_random_direction(current_point, g);
            temp_step = find_max_step(algs, g, current_point, temp_direction, pagerank, temp_loss_function_value, current_loss_function_value);
            // Check if it is better than the candidate point.
            if (temp_loss_function_value < candidate_loss_value) {
                candidate_loss_value = temp_loss_function_value;
                candidate_direction = get_step_direction(temp_step, temp_direction,number_of_nodes);
                get_candidate_point(current_point, candidate_direction, candidate_point, number_of_nodes);
            }
        }
        // Renew values.
        local_log_file << "End of direction searching.\nI was in point: [";
        local_log_file << "\n";
        local_log_file << "With loss value: " << current_loss_function_value << "\n";
        current_point = candidate_point;
        current_loss_function_value = candidate_loss_value;
        local_log_file << "I will go to point: [";
        local_log_file << "\n";
        local_log_file << "With loss value: " << current_loss_function_value << "\n";
        local_log_file << "Continue searching to improve new current point if i have functional calculations left\n"; 
    }
    local_log_file << "End of functional calculations - end of Searching.\n";
    // Renew values.
    if (current_loss_function_value < best_loss_value) {
        best_point = current_point;
        best_loss_value = current_loss_function_value;
    }

    local_log_file << "current loss value: " << current_loss_function_value << "\n";
    local_log_file << "best loss value: " << best_loss_value << "\n";
    local_log_file.close();
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, best_point);
    temp_fair_pagerank = algs.get_local_fair_pagerank();
    save_results(temp_fair_pagerank, best_point);
    std::cout << "red ratio: " << g.get_pagerank_per_community(temp_fair_pagerank)[1];

    return 0;
}