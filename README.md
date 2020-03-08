# FairLaR
Fairness-Aware Link Analysis

<!--![Datasets feture image](/img/datasets_header.jpg)<br/>-->
![Datasets feture image](https://lh5.googleusercontent.com/proxy/7bE3tvuQMnA--9SwvI2cFawze2U5JdNIvT5I90qfEC6A0uO3ENc0sCrDkrSnD3ikv0KVXbH-HcvL3a3gv2_TKjhixnOVVft7JZlNpODTVXJXsnK63JAEU8pkieRf)<br/>

General Informations.
---------------------
This repository has been created to distribute freely our implementation of the algorithms described at the "Fairness-Aware Link Analysis" paper. It also includes the datasets described and used in the forthmentioned paper plus some extra dataset from various resources.

All the experiments were made in linux UBUNTU.

### Repository Structure:<br/>
- FairLar
    - Code
        - Cpp_files
        - Python_files
    - Datasets

### How to run the algorithms:<br/>  

1. Generate synthetic:
    
    For this task we use a forest fire model properly modified to take into consideration the homophily of each group. To generate synthetic dataset you must provide 6 parameters. Homopihly of group 0, homophily of group 1, probability to belong to protected group (group 1), number of nodes, edges per node, number of initial nodes. Homophily for each group is a float number in [0, 1] which id the probability to connect with a node of the same group.

    >`./homophily_graph_generator.out 0.2 0.4 0.3 1000 5 3`    

2. Pagerank, LFPR_N, LFPR_U, LFPR_P:

    You should run the **`./pagerank.out`** inside the dataset's folder, which should contain the "out_graph.txt" file and the "out_community.txt" file. Structure of these files described [here](#datasets-description). You have the following options:

    > `./pagerank.out` : Run the forth mentioned algorithms algorithms with phi = ratio of protected category.

    > `./pagerank.out -c <community_file>` : Run the forth mentioned algorithms with phi described in <community_file>. Community file is a txt file in which id line is a pair of an integer and a float separated with empty space. The integer is the id of the group while the float is tha wanted ratio for this group. There should be as many lines as the groups of the nodes and the sum of the floats should be equal to 1. (e.g. 0 0.3 \n 1 0.7)

    > `./pagerank.out -pn <node>` : Run the forthmentioned algorithms in personilized - by node - mode. \<node\> is an integer - the id of the node.

    > `.pagerank.out -tk <k>/` : Run the forthmetnioned algorithms in targeted mode for the top-k nodes by pagerank. \<k\> is an integer denotes the number of the nodes that will be taken into consideration.

    You can also use the following easy to understand commands. 

    > `./pagerank.out -pn <node> -c <community_file>`

    > `./pagerank.out -tk <k> -c <community_file>`

3. Personilized by node pagerank for all nodes of a network:

    You should run the **`./person_all_nodes.out`** inside the dataset's folder, which should contain the "out_graph.txt" file and the "out_community.txt" file.

    > `./person_all_nodes.out`

4. Sensitive:

    You should run the **`./sensitive.py`** inside the dataset's folder, which should contain the "out_graph.txt" file and the "out_community.txt" file. You should also provide phi = wanted ratio for the protected group. If phi = 0, then phi = ratio of protected group.

    >`python sensitive.py 0.5`

5. Targeted, Targeted optimal, targeted top-k:

    You should run each one of these algorithms inside the dataset's folder, which should contain the "out_graph.txt" file and the "out_community.txt" file. You should also provide phi = wanted ratio for the protected group and k = number of nodes for the targeted algorithms. If phi = 0, then phi = ratio of protected group.

    > `python targeted.py 0.5 10`

    > `python targeted_optimal.py 0.5 10`
    
    > `python targeted_topk.py 0.5 10`

Datasets-Description.
---------
