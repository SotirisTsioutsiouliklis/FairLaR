import numpy as np
import sys
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def local_blox_plots():

    algorithms = ["lfprn" , "lfpru", "lfprp"]
    folders = ["books", "blogs", "dblp_course", "dblp_spyros", "twitter", "tmdb"]

    up_red = dict()
    up_blue = dict()
    down_red = dict()
    down_blue = dict()

    for algo in algorithms:
        # Init lists.
        up_red[algo] = []
        up_blue[algo] = []
        down_red[algo] = []
        down_blue[algo] = []

        # Add values to lists.
        for folder in folders:
            with open("%s/out_%s_value_dif_weighted.txt" %(folder, algo), "r") as file_one:
                file_one.readline()
                j = 0
                for line in file_one:
                    cat = int(line.split()[2])
                    value = float(line.split()[7])
                    if j < 10:
                        if cat == 1:
                            down_red[algo].append(value)
                        else:
                            down_blue[algo].append(value)
                    else:
                        if cat == 1:
                            up_red[algo].append(value)
                        else:
                            up_blue[algo].append(value)
                    j += 1

    # Transform to numpy array.
    for algo in algorithms:
        # Init lists.
        up_red[algo] = np.array(up_red[algo])
        up_blue[algo] = np.array(up_blue[algo])
        down_red[algo] = np.array(down_red[algo])
        down_blue[algo] = np.array(down_blue[algo])

    plt.rcParams['xtick.labelsize'] = 25
    # Plot boxplots.
    fig = plt.figure(figsize=(35,25))
    plt.ylabel("Weight Ratio of Protected Group", fontsize=25)
    #plt.title("Nodes that lose value")
    plt.rcParams['xtick.labelsize'] = 25
    plt.boxplot([down_red["lfprn"], down_blue["lfprn"], down_red["lfpru"], down_blue["lfpru"], down_red["lfprp"], down_blue["lfprp"]], notch= True, showfliers=False, 
                labels=["Red Group LFPR_N", "Blue Group LFPR_N", "Red Group LFPR_U", "Blue Group LFPR_U", "Red Group LFPR_P", "Blue Group LFPR_P"])
    plt.xticks(rotation="-45")
    plt.savefig("local_value_loss_box_plots.pdf")
    plt.savefig("local_value_loss_box_plots.png")

    '''
    fig = plt.figure(figsize=(20,12))
    plt.ylabel("Weight Ratio of Protected Group")
    #plt.title("Nodes that gain value")
    plt.boxplot([up_red["lfprn"], up_blue["lfprn"], up_red["lfpru"], up_blue["lfpru"], up_red["lfprp"], up_blue["lfprp"]], notch= False, showfliers=False, 
                labels=["Red Group LFPR_N", "Blue Group LFPR_N", "Red Group LFPR_U", "Blue Group LFPR_U", "Red Group LFPR_P", "Blue Group LFPR_P"])
    plt.xticks(rotation="-45")
    plt.savefig("local_gainValue_box_plots_1.pdf")
    plt.savefig("local_gainValue_box_plots_1.png")
    '''

    fig = plt.figure(figsize=(35,25))
    plt.ylabel("Weight Ratio of Protected Group", fontsize=25)
    plt.rcParams['xtick.labelsize'] = 25
    #plt.title("Nodes that gain value")
    plt.boxplot([up_red["lfprn"], up_blue["lfprn"], up_red["lfpru"], up_blue["lfpru"], up_red["lfprp"], up_blue["lfprp"]], notch= True, showfliers=False, 
                labels=["Red Group LFPR_N", "Blue Group LFPR_N", "Red Group LFPR_U", "Blue Group LFPR_U", "Red Group LFPR_P", "Blue Group LFPR_P"])
    plt.xticks(rotation="-45")
    plt.savefig("local_gainValue_box_plots_2.pdf")
    plt.savefig("local_gainValue__box_plots_2.png")

def sensitive_box_plot():
    algorithms = ["sensitive"]
    folders = ["karate", "books", "blogs", "dblp_course", "dblp_spyros", "physics", "twitter", "tmdb", "github_fem", "github_male"]

    up_red = dict()
    up_blue = dict()
    down_red = dict()
    down_blue = dict()

    for algo in algorithms:
        # Init lists.
        up_red[algo] = []
        up_blue[algo] = []
        down_red[algo] = []
        down_blue[algo] = []

        # Add values to lists.
        for folder in folders:
            with open("%s/out_%s_value_dif_weighted.txt" %(folder, algo), "r") as file_one:
                file_one.readline()
                j = 0
                for line in file_one:
                    cat = int(line.split()[2])
                    value = float(line.split()[7])
                    if j < 10:
                        if cat == 1:
                            down_red[algo].append(value)
                        else:
                            down_blue[algo].append(value)
                    else:
                        if cat == 1:
                            up_red[algo].append(value)
                        else:
                            up_blue[algo].append(value)
                    j += 1

    # Transform to numpy array.
    for algo in algorithms:
        # Init lists.
        up_red[algo] = np.array(up_red[algo])
        up_blue[algo] = np.array(up_blue[algo])
        down_red[algo] = np.array(down_red[algo])
        down_blue[algo] = np.array(down_blue[algo])

    plt.rcParams['xtick.labelsize'] = 25
    # Plot boxplots.
    fig = plt.figure(figsize=(35,25))
    plt.ylabel("Weight Ratio of Protected Group", fontsize=25)
    #plt.title("Nodes that lose value")
    plt.rcParams['xtick.labelsize'] = 25
    plt.boxplot([down_red["sensitive"], down_blue["sensitive"]], notch= True, showfliers=False, 
                labels=["Red Group sensitive", "Blue Group sensitive"])
    plt.xticks(rotation="-45")
    plt.savefig("sensitive_value_loss_box_plots.pdf")
    plt.savefig("sensitive_value_loss_box_plots.png")

    '''
    fig = plt.figure(figsize=(20,12))
    plt.ylabel("Weight Ratio of Protected Group")
    #plt.title("Nodes that gain value")
    plt.boxplot([up_red["lfprn"], up_blue["lfprn"], up_red["lfpru"], up_blue["lfpru"], up_red["lfprp"], up_blue["lfprp"]], notch= False, showfliers=False, 
                labels=["Red Group LFPR_N", "Blue Group LFPR_N", "Red Group LFPR_U", "Blue Group LFPR_U", "Red Group LFPR_P", "Blue Group LFPR_P"])
    plt.xticks(rotation="-45")
    plt.savefig("local_gainValue_box_plots_1.pdf")
    plt.savefig("local_gainValue_box_plots_1.png")
    '''

    fig = plt.figure(figsize=(35,25))
    plt.ylabel("Weight Ratio of Protected Group", fontsize=25)
    plt.rcParams['xtick.labelsize'] = 25
    #plt.title("Nodes that gain value")
    plt.boxplot([up_red["sensitive"], up_blue["sensitive"]], notch= True, showfliers=False, 
                labels=["Red Group sensitive", "Blue Group sensitive"])
    plt.xticks(rotation="-45")
    plt.savefig("sensitive_gainValue_box_plots_2.pdf")
    plt.savefig("sensitive_gainValue__box_plots_2.png")

#sensitive_box_plot()
local_blox_plots()