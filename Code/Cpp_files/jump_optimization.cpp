#include <fstream>
#include "jump_optimization.hpp"
#include <omp.h>

// Parameters
int FUNCTIONAL_CALCULATIONS = 10000;
int NUMBER_OF_DIRECTIONS = 100;
double PRECISION_OF_SOLUTION = pow(10, -8);
double PRECISION_OF_RED_RATIO = pow(10, -4);
int TIMES_TO_TRY_FOR_INITIAL_POINT = 100;
int iter = 0;
//std::ofstream jump_log_file;

// Save results.
void save_results(pagerank_v &fair_pagerank, std::vector<double> &jump_vector) {
    std::ofstream pagerank_file;
    pagerank_file.open("opt_mine_fairpagerank_vector.txt");
    std::ofstream vector_file;
    vector_file.open("opt_mine_jump_vector.txt");

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
std::vector<double> get_random_initial_point(pagerank_algorithms &algs, graph &g, double phi) {
    int dimension = g.get_num_nodes();
    std::vector<double> initial_point(dimension, 0);
    bool found_smaller_than_phi = false;
    bool found_greater_than_phi = false;
    bool found_feasible_vector = false;
    int smaller_phi_node;
    int greater_phi_node;
    double smaller_phi;
    double greater_phi;
    std::vector<int> nodes(dimension);
    for (int i = 0; i < dimension; i ++) {
        nodes[i] = i;
    }
    std::random_shuffle(nodes.begin(), nodes.end());

    int node_order = 0;
    while ((!found_feasible_vector) && (node_order < TIMES_TO_TRY_FOR_INITIAL_POINT)) {
        int node = nodes[node_order];
        node_order ++; 
        algs.set_personalization_type(personalization_t::NODE_PERSONALIZATION, node);
        pagerank_v personalize_pagerank = algs.get_pagerank();
        iter++;
        double red_ratio = g.get_pagerank_per_community(personalize_pagerank)[1];
        if ((red_ratio < phi) && !found_smaller_than_phi) {
            found_smaller_than_phi = true;
            smaller_phi_node = node;
            smaller_phi = red_ratio;
            //std::cout << "Smaller phi initialized\n";
        } else if ((red_ratio > phi) && !found_greater_than_phi) {
            found_greater_than_phi = true;
            greater_phi_node = node;
            greater_phi = red_ratio;
            //std::cout << "Greater phi initialized\n";
        } 
    
        found_feasible_vector = found_greater_than_phi && found_smaller_than_phi;
    }
    //std::cout << "Found init point\n";
    //std::cout << "Times to find it: " << node_order << std::endl;
    if (node_order == TIMES_TO_TRY_FOR_INITIAL_POINT) {
        exit(1 );
    }

    // c:= coefficient.
    // Solve: c*greater_phi + (1-c) * smaller_phi == phi.
    double coefficient = (double)(phi - smaller_phi) / (greater_phi - smaller_phi);
    initial_point[greater_phi_node] = coefficient;
    initial_point[smaller_phi_node] = 1 - coefficient;

    return initial_point;
}

// Returns valide direction.
std::vector<double> create_random_direction(std::vector<double> &current_point, int dimension) {
    std::vector<double> direction(dimension, 0);
    std::vector<double> temp_point(dimension);
    std::vector<int> nodes(dimension);
    double quantity_to_add = 0;
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
            quantity_to_add += quantity_to_take;
        }
    }
  
    std::random_shuffle(nodes.begin(), nodes.end());   
    get_candidate_point(current_point, direction, temp_point, dimension);

    int i = 0;
    while (quantity_to_add > pow(10, -4)) {
        node = nodes[i];
        max_to_give = std::min((1-temp_point[i]), quantity_to_add);
        quantity_to_give = ((double)rand() / RAND_MAX) * max_to_give;
        direction[node] += quantity_to_give;
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
            direction[node] += quantity_to_add;
            quantity_to_add = 0;
            break;
        }
    }

    return direction;
}

double find_max_step(pagerank_algorithms &algs , graph &g, std::vector<double> &current_point, std::vector<double> &direction,
     pagerank_v &pagerank, double &temp_loss_function_value, double &current_loss_function_value, double &phi) 
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
    temp_pagerank = algs.get_pagerank(); 
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
        temp_pagerank = algs.get_pagerank(); 
        iter++;
        temp_loss_function_value = loss_function_at(pagerank, temp_pagerank, dimension);
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
        temp_pagerank = algs.get_pagerank();
    }
    
    
    // Find fair result.
    while (!is_fair_pagerank(temp_pagerank, g, phi)) {
        step = step/(double)2;
        step_direction = get_step_direction(step, direction, dimension);
        get_candidate_point(current_point, step_direction, point, dimension);
        algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, point);
        temp_pagerank = algs.get_pagerank();
        iter++;
        if (abs(step) < pow(10, -10)) {
            step = 0;
            step_direction = get_step_direction(step, direction, dimension);
            get_candidate_point(current_point, step_direction, point, dimension);
            return step;
        }
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

double soft_loss_function_at(pagerank_v &pagerank, pagerank_v &temp_fair_pagerank, graph &g, double phi) {
    int dimension = g.get_num_nodes();
    double sum = 0;
    for (int i = 0; i < dimension; i++) {
        sum += pow((pagerank[i].pagerank - temp_fair_pagerank[i].pagerank), 2);
    }

    double temp_red_ratio = g.get_pagerank_per_community(temp_fair_pagerank)[1];

    double diff = abs(temp_red_ratio - phi);
    sum += std::tanh(diff);

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
    //omp_set_num_threads(20);
    // Read command line arguments.
    double phi;
    
    if (argc == 1) {
		std::cout << "Provide <phi: wanted ratio for category 1>\n";
        return 0;
	} else if (argc == 2) {
        phi = std::atof(argv[1]);
    } else {
		std::cout << "Provide ONLY ONE ARGUMENT <phi: wanted ratio for category 1>\n";
        return 0;
    }
    

    
    graph g("out_graph.txt", "out_community.txt"); // Load graph.
    if (phi == 0) phi = g.get_community_percentage(1);// ----------------------------------------------------------------------------------------------
    // Create phi file which is needed for personilized pagerank.
    std::ofstream phi_file;
    phi_file.open("out_phi.txt");
    phi_file << "0\t" << 1 - phi << "\n";
    phi_file << "1\t" << phi;
    phi_file.close();

    //jump_log_file.open("jump_log_file.txt");
    //jump_log_file << "Start experiments with phi: " << phi << "\n";
    // Initializations.
    srand(time(NULL)); // Init randomly
    //////std::cout << "read graph\n";
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
    current_point = get_random_initial_point(algs, g, phi);
    //jump_log_file << "Initial point: [";
    //for (int l = 0; l < number_of_nodes; l++) {
        //if (current_point[l] != 0) {
            //jump_log_file << l << " : " << current_point[l] << "\n";
        //}
        //jump_log_file  << " " << current_point[l] << ",";
    //}
    //jump_log_file << "\n";
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, current_point);
    current_pagerank = algs.get_pagerank();
    iter++;
    current_loss_function_value = loss_function_at(pagerank, current_pagerank, number_of_nodes);
    candidate_loss_value = current_loss_function_value;

    if (current_loss_function_value < best_loss_value) {
        best_point = current_point;
        best_loss_value = current_loss_function_value;
    }
    
    //jump_log_file << "current loss value: " << current_loss_function_value << "\n";
    //jump_log_file << "best loss value: " << best_loss_value << "\n";  
    //jump_log_file << "Start iterations:\n";
    while (iter < FUNCTIONAL_CALCULATIONS) {
        whole_iterations++;
        std::cout << "New point\n";
        //jump_log_file << "Iteration: " << whole_iterations << "\n";
        
        if (restart_condition(best_loss_value, current_loss_function_value)) {
            //jump_log_file << "-------------------------------------------RESTART----------------------------------------\n";
            if (current_loss_function_value < best_loss_value) {
            best_point = current_point;
            best_loss_value = current_loss_function_value;
            }
            //jump_log_file << "current loss value: " << current_loss_function_value << "\n";
            //jump_log_file << "best loss value: " << best_loss_value << "\n";
            goto restart;
        }
        
        // Find the best direction and corresponding step.
        // current_loss_function_value = loss_function_value;
        for (int i = 0; i < NUMBER_OF_DIRECTIONS; i++) {
            std::cout << "New direction\n";
            temp_direction = create_random_direction(current_point, number_of_nodes);
            temp_step = find_max_step(algs, g, current_point, temp_direction, pagerank, temp_loss_function_value, current_loss_function_value, phi);
            // Check if it is better than the candidate point.
            if (temp_loss_function_value < candidate_loss_value) {
                candidate_loss_value = temp_loss_function_value;
                candidate_direction = get_step_direction(temp_step, temp_direction,number_of_nodes);
                get_candidate_point(current_point, candidate_direction, candidate_point, number_of_nodes);
            }
        }
        // Renew values.
        //jump_log_file << "End of direction searching.\nI was in point: [";
        //jump_log_file << "\n";
        //jump_log_file << "With loss value: " << current_loss_function_value << "\n";
        current_point = candidate_point;
        current_loss_function_value = candidate_loss_value;
        //jump_log_file << "I will go to point: [";
        //jump_log_file << "\n";
        //jump_log_file << "With loss value: " << current_loss_function_value << "\n";
        //jump_log_file << "Continue searching to improve new current point if i have functional calculations left\n"; 
    }
    //jump_log_file << "End of functional calculations - end of Searching.\n";
    // Renew values.
    if (current_loss_function_value < best_loss_value) {
        best_point = current_point;
        best_loss_value = current_loss_function_value;
    }

    //jump_log_file << "current loss value: " << current_loss_function_value << "\n";
    //jump_log_file << "best loss value: " << best_loss_value << "\n";
    //jump_log_file.close();
    algs.set_personalization_type(personalization_t::JUMP_OPT_PERSONALIZATION, best_point);
    temp_fair_pagerank = algs.get_pagerank();
    save_results(temp_fair_pagerank, best_point);
    std::cout << "red ratio: " << g.get_pagerank_per_community(temp_fair_pagerank)[1];

    return 0;
}