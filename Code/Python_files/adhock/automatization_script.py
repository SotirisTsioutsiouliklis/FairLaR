import subprocess

small_datasets = ["books", "blogs", "dblp_spyros", "tmdb"]
big_datasets = ["dblp_course", "twitter", "physics"]
phi = [0.3, 0.5, 0.7]

def copy_executables():
    for d in small_datasets:
        subprocess.call(["cp ../../FairLaR/Code/Cpp_files/pagerank.out %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Cpp_files/residual_optimization.out %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Cpp_files/jump_optimization.out %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Python_files/sensitive.py %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Python_files/excess_opt_2.py %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Python_files/excess_opt.py %s" %d], cwd=".", shell=True)
    for d in big_datasets:
        subprocess.call(["cp ../../FairLaR/Code/Cpp_files/pagerank.out %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Cpp_files/residual_optimization.out %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Cpp_files/jump_optimization.out %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Python_files/sensitive.py %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Python_files/excess_opt_2.py %s" %d], cwd=".", shell=True)
        subprocess.call(["cp ../../FairLaR/Code/Python_files/excess_opt.py %s" %d], cwd=".", shell=True)

def run_experiments():
    for f in phi:
        subprocess.call(["mkdir %.2f" %f], cwd=".", shell=True)
        subprocess.call(["cp -r * %.2f" %f], cwd=".", shell=True)
        for d in small_datasets:
            print("------------ %s -----------" %d)
            # Run pagerank.
            subprocess.call(["./pagerank.out"], cwd="%.2f/%s" %(f, d), shell=True)
            # Run jump optimizations.
            subprocess.call(["python3 sensitive.py %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            subprocess.call(["./jump_optimization.out %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            # Run excess optimizations.
            subprocess.call(["python3 excess_opt.py %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            subprocess.call(["python3 excess_opt_2.py %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            subprocess.call(["residual_optimization.out %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
    for f in phi:
        subprocess.call(["cp -r * %.2f" %f], cwd=".", shell=True)
        for d in big_datasets:
            print("------------ %s -----------" %d)
            # Run pagerank.
            subprocess.call(["./pagerank.out"], cwd="%.2f/%s" %(f, d), shell=True)
            # Run jump optimizations.
            subprocess.call(["python3 sensitive.py %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            subprocess.call(["./jump_optimization.out %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            # Run excess optimizations.
            subprocess.call(["python3 excess_opt.py %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            subprocess.call(["python3 excess_opt_2.py %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)
            subprocess.call(["residual_optimization.out %.2f" %f], cwd="%.2f/%s" %(f, d), shell=True)

copy_executables()
run_experiments()