# Very simple Makefile to compile everything

all:
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o residual_optimization.out graph.cpp pagerank.cpp residual_stochastic_opt.cpp
	g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o pagerank.out graph.cpp pagerank.cpp pagerank_driver.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o person_all_nodes.out graph.cpp pagerank.cpp person_all_nodes.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o data_preprocessor.out data_preprocessor.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o homophily_graph_generator.out synthetic_graph_generator.cpp

clean:
	#rm -f jump_optimization