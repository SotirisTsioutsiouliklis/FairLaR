import subprocess

datasets = ["books", "blogs", "dblp_course", "twitter", "physics", "dblp_aminer", "pokec", "linkedin"]

phis = [0, 0.5, 0.3, 0.7]

for phi in phis:
    # Create phi file.
    with open("out_phi.txt", "w") as file_one:
        file_one.write("0 %.2f" %(1-phi))
        file_one.write("1 %.2f" %phi)
    # Create the phi folder.
    subprocess.call("mkdir phi_%.2f" %phi, cwd=".", shell=True)
    for d in datasets:
        print("---------------%f , %s ----------------" %(phi, d))
        # Copy Datasets in phi folder.
        subprocess.call("cp -r %s phi_%.2f/" %(d, phi), cwd=".", shell=True)
        subprocess.call("cp out_phi.txt phi_%.2f/%s" %(phi, d), cwd=".", shell=True)
        # Copy executables.
        subprocess.call("cp pagerank.out phi_%.2f/%s/" %(phi, d), cwd=".", shell=True)
        # Run executables.
        if phi == 0:
            subprocess.call("./pagerank.out", cwd="phi_%.2f/%s" %(phi, d), shell=True)
        else:
            subprocess.call("./pagerank.out -c out_phi.txt", cwd="phi_%.2f/%s" %(phi, d), shell=True)