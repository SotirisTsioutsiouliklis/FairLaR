# Very simple Makefile to compile everything

all:
	g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o pagerank graph.cpp pagerank.cpp pagerank_driver.cpp
	g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o excess_optimization.out graph.cpp pagerank.cpp excess_optimization.cpp
	g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o person_all_nodes graph.cpp pagerank.cpp person_all_nodes.cpp
	g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o fairout graph.cpp pagerank.cpp fairout.cpp
	g++ --std=c++11 -Wall -Wextra -O3 -o data_preprocessor data_preprocessor.cpp
	g++ --std=c++11 -Wall -Wextra -O3 -o homophily_graph_generator homophily_graph_generator.cpp

clean:
	rm -f pagerank data_preprocessor