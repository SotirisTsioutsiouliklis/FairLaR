#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

// NOTE: before process_nodes() is called: key = old_id, value = 0 if this
// node does not participate in any edge and 1 if it does.  After
// process_nodes() is called: key = old_id, value = new_id
static std::unordered_map<int, int> node_map;
// NOTE: before process_communities() is called: key = old_id, value =
// number of nodes with this community.  After process_communities() is
// called: key = old_id, value = new_id
static std::unordered_map<int, int> community_map;
static std::unordered_map<int, int> attribute_map; // key = old_id, value = new_id
static std::unordered_set<int> nodes_to_delete;
static std::unordered_set<int> communities_to_delete;

// version 0: nodes without community get thrown away
// version 1: nodes without community get assigned community 0
static bool run_version_0;
// these nodes must be added in community file if we run in version 1
static std::vector<int> nodes_without_community;

static void add_node(const int node_id)
{
	node_map.insert({node_id, 0});
}

static void add_community(const int community_id)
{
	if (community_map.find(community_id) == community_map.end())
		community_map.insert({community_id, 1});
	else
		++community_map[community_id];
}

static void add_attribute(const int node_id, const int attribute_id)
{
	if (node_map.find(node_id) != node_map.end())
		attribute_map.insert({attribute_id, 0});
}

static int get_num_nodes()
{
	return node_map.size();
}

static int get_num_communities()
{
	return community_map.size();
}

static int get_num_attributes()
{
	return attribute_map.size();
}

static void process_nodes()
{
	for (auto it = node_map.begin(); it != node_map.end(); ++it)
		if (it->second == 0)
			nodes_to_delete.insert(it->first);

	for (const auto &node : nodes_to_delete)
		node_map.erase(node_map.find(node));

	int i = 0;
	for (auto it = node_map.begin(); it != node_map.end(); ++it)
		it->second = i++;
}

static void process_communities(std::string &in_filename)
{
	std::ifstream infile(in_filename);
	std::string line;

	while (std::getline(infile, line))
	{
		// get rid of comments
		auto pos = line.find("#");
		if (pos != std::string::npos)
			line.erase(line.begin() + pos, line.end());

		std::istringstream line_stream(line);

		int value1, value2;
		while (line_stream >> value1 >> value2)
			if (nodes_to_delete.find(value1) != nodes_to_delete.end())
				--community_map[value2];
	}
	infile.close();

	for (auto it = community_map.begin(); it != community_map.end(); ++it)
		if (it->second == 0)
			communities_to_delete.insert(it->first);

	for (const auto &community : communities_to_delete)
		community_map.erase(community_map.find(community));

	int i = 0;
	for (auto it = community_map.begin(); it != community_map.end(); ++it)
		if (get_num_communities() == 2)
			// do not change the community assignment in order not
			// to mess Julia's metric (rND)
			it->second = it->first;
		else
			it->second = i++;
}

static void process_attributes()
{
	int i = 0;
	for (auto it = attribute_map.begin(); it != attribute_map.end(); ++it)
		it->second = i++;
}

static int get_new_node_id(const int old_node_id)
{
	return node_map[old_node_id];
}

static int get_new_community_id(const int old_community_id)
{
	return community_map[old_community_id];
}

static int get_new_attribute_id(const int old_attribute_id)
{
	return attribute_map[old_attribute_id];
}

static void set_used_node(const int node1, const int node2)
{
	if (run_version_0)
	{
		if ((node_map.find(node1) != node_map.end()) && (node_map.find(node2) != node_map.end()))
			node_map[node1] = node_map[node2] = 1;
	}
	else // version 1
	{
		if (node_map.find(node1) == node_map.end())
		{
			nodes_without_community.push_back(node1);
			add_community(0); // another node with community 0
		}

		if (node_map.find(node2) == node_map.end())
		{
			nodes_without_community.push_back(node2);
			add_community(0); // another node with community 0
		}

		node_map[node1] = node_map[node2] = 1;
	}
}

static void load_file(const std::string &in_filename, const bool is_nodes_file)
{
	std::ifstream infile(in_filename);
	std::string line;

	while (std::getline(infile, line))
	{
		// get rid of comments
		auto pos = line.find("#");
		if (pos != std::string::npos)
			line.erase(line.begin() + pos, line.end());

		std::istringstream line_stream(line);

		int value1, value2;
		while (line_stream >> value1 >> value2)
			if (is_nodes_file)
			{
				set_used_node(value1, value2);
			}
			else // communities file
			{
				add_node(value1);
				add_community(value2);
			}
	}

	infile.close();
}

static void load_attributes(char *in_filename)
{
	std::ifstream infile(in_filename);
	std::string line;

	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		int node, attribute;

		iss >> node;
		while (iss >> attribute)
		{
			add_attribute(node, attribute);
		}
	}

	infile.close();
}

static void save_file(const std::string &in_filename, const std::string &out_filename,
		const bool is_nodes_file)
{
	std::ifstream infile(in_filename);
	std::string line;

	std::ofstream outfile(out_filename);
	outfile << ((is_nodes_file) ? get_num_nodes() : get_num_communities()) << std::endl;

	while (std::getline(infile, line))
	{
		// get rid of comments
		auto pos = line.find("#");
		if (pos != std::string::npos)
			line.erase(line.begin() + pos, line.end());

		std::istringstream line_stream(line);

		int value1, value2;
		while (line_stream >> value1 >> value2)
			if (is_nodes_file)
			{
				if (node_map.find(value1) != node_map.end() &&
						node_map.find(value2) != node_map.end())
					outfile << get_new_node_id(value1) << " " << get_new_node_id(value2) << std::endl;
			}
			else // communities file
				if (node_map.find(value1) != node_map.end())
					outfile << get_new_node_id(value1) << " " << get_new_community_id(value2) << std::endl;
	}

	if (!is_nodes_file && !run_version_0)
	{
		for (const auto &node : nodes_without_community)
			outfile << get_new_node_id(node) << " " << 0 << std::endl; // XXX if we rename communities we have to change this!
	}

	infile.close();
	outfile.close();
}

static void save_attributes(char *in_filename, std::string out_filename, std::string attr_transf_filename)
{
	std::ifstream infile(in_filename);
	std::string line;

	std::ofstream outfile(out_filename);
	outfile << get_num_attributes() << std::endl;

	std::ofstream outfile_transformation(attr_transf_filename);

	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		int node, attribute;

		iss >> node;
		if (node_map.find(node) == node_map.end())
			continue; // node not found

		outfile << get_new_node_id(node);
		while (iss >> attribute)
		{
			outfile << " " << get_new_attribute_id(attribute);
		}
		outfile << std::endl;
	}

	for (auto it = attribute_map.cbegin(); it != attribute_map.cend(); ++it)
		outfile_transformation << it->first << " => " << it->second << std::endl;

	infile.close();
	outfile.close();
	outfile_transformation.close();
}

static void save_node_transformation(std::string out_filename)
{
	std::ofstream outfile(out_filename);

	for (auto it = node_map.cbegin(); it != node_map.cend(); ++it)
		outfile << it->first << " => " << it->second << std::endl;

	outfile.close();
}

static void save_community_transformation(std::string out_filename)
{
	std::ofstream outfile(out_filename);

	for (auto it = community_map.cbegin(); it != community_map.cend(); ++it)
		outfile << it->first << " => " << it->second << std::endl;

	outfile.close();
}

int main(int argc, char *argv[])
{
	if (((argc != 2) && (argc != 3)) ||
			(std::strcmp(argv[1], "0") && std::strcmp(argv[1], "1")))
	{
		std::cerr << "Usage: " << argv[0] << " 0|1 [input_personalization]\n"
			"\twhere 0 is Version 0 and 1 is Version 1" << std::endl;
		return 1;
	}
	const bool run_personalized = (argc == 3);
	run_version_0 = (std::strcmp(argv[1], "0") == 0);

	std::string ingraph("in_graph.txt");
	std::string incommunities("in_community.txt");
	char cmd[24]; sprintf(cmd, "mkdir version%d", !run_version_0);
	char outgraph [48]; sprintf(outgraph, "version%d/out_graph.txt", !run_version_0);
	char outcommunities [48]; sprintf(outcommunities, "version%d/out_community.txt", !run_version_0);
	char outperson [48]; sprintf(outperson, "version%d/out_person.txt", !run_version_0);
	char node_transf_filename [48]; sprintf(node_transf_filename, "version%d/node_transformation.txt", !run_version_0);
	char comm_transf_filename [48]; sprintf(comm_transf_filename, "version%d/community_transformation.txt", !run_version_0);
	char attr_transf_filename [48]; sprintf(attr_transf_filename, "version%d/attribute_transformation.txt", !run_version_0);

	// C++ does not have a standard way to create directories,
	// but the following works on both Windows and Linux
	std::system(cmd);

	load_file(incommunities, false);
	load_file(ingraph, true);

	process_nodes();
	process_communities(incommunities);

	save_file(incommunities, outcommunities, false);
	save_file(ingraph, outgraph, true);

	save_node_transformation(node_transf_filename);
	save_community_transformation(comm_transf_filename);

	if (run_personalized) // we deal with personalization after processing the nodes, ...
	{                     // ... so we are sure that all the nodes are valid now
		load_attributes(argv[2]);
		process_attributes();
		save_attributes(argv[2], outperson, attr_transf_filename);
	}
}
