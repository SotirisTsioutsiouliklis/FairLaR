# Very simple Makefile to compile everything

all:
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o experimental graph.cpp pagerank.cpp experimental.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o jump_optimization graph.cpp pagerank.cpp jump_optimization.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o local_optimization graph.cpp pagerank.cpp local_optimization.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o excess_optimization.out graph.cpp pagerank.cpp excess_optimization.cpp
	g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o stochastic_optimization.out graph.cpp pagerank.cpp residual_stochastic_opt.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o pagerank graph.cpp pagerank.cpp pagerank_driver.cpp
	#g++ --std=c++11 -Wall -Wextra -O3 -fopenmp -o testing graph.cpp pagerank.cpp testing.cpp

clean:
	rm -f jump_optimization