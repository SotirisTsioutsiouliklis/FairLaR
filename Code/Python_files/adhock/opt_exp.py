import subprocess

big_datasets = ["physics", "dblp_aminer", "pokec", "linkedin"]


for d in big_datasets:
    print("--------------- %s ----------------" % d)
    subprocess.call("./residual_optimization.out 0.5", cwd="%s" %d, shell=True)
