import subprocess

small_datasets = ["books", "blogs", "dblp_course", "twitter"]

big_datasets = ["dblp_aminer", "pokec", "linkedin", "physics"]

small_phis = [0.5]
big_phis = [0, 0.5] 

for phi in small_phis:
    # Create the phi folder.
    subprocess.call("mkdir phi_%.2f" %phi, cwd=".", shell=True)
    for d in small_datasets:
        print("---------------%f , %s ----------------" %(phi, d))
        # Copy Datasets in phi folder.
        subprocess.call("cp -r %s phi_%.2f/" %(d, phi), cwd=".", shell=True)
        # Copy executables.
        subprocess.call("cp pagerank.out phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        subprocess.call("cp residual_optimization.out phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        subprocess.call("cp excess_opt.py phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        subprocess.call("cp compare_opt_algorithms.py phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        # Run executables.
        subprocess.call("./residual_optimization.out %.2f" %phi, cwd="phi_%.2f/%s" %(phi, d), shell=True)
        subprocess.call("./pagerank.out -c out_phi.txt", cwd="phi_%.2f/%s" %(phi, d), shell=True)
        subprocess.call("python3 excess_opt.py %.2f" %phi, cwd="phi_%.2f/%s" %(phi, d), shell=True)

for phi in big_phis:
    # Create the phi folder.
    subprocess.call("mkdir big_phi_%.2f" %phi, cwd=".", shell=True)
    for d in big_datasets:
        print("---------------%f , %s ----------------" %(phi, d))
        # Copy Datasets in phi folder.
        subprocess.call("cp -r %s big_phi_%.2f/" %(d, phi), cwd=".", shell=True)
        # Copy executables.
        subprocess.call("cp pagerank.out big_phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        subprocess.call("cp residual_optimization.out big_phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        #subprocess.call("cp excess_opt.py big_phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        subprocess.call("cp compare_opt_algorithms.py big_phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        # Run executables.
        subprocess.call("./residual_optimization.out %.2f" %phi, cwd="big_phi_%.2f/%s" %(phi, d), shell=True)
        subprocess.call("./pagerank.out -c out_phi.txt", cwd="big_phi_%.2f/%s" %(phi, d), shell=True)
        #subprocess.call("python3 excess_opt.py %.2f" %phi, cwd="big_phi_%.2f/%s" %(phi, d), shell=True)

