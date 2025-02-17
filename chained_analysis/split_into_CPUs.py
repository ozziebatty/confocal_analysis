import multiprocessing
import time
from multiprocessing import current_process
import numpy as np

def worker_function(file_to_process):
    """
    Function that will run on available CPUs.
    Processes an element and prints the process name/ID.
    """
    # Get the current process info
    process = current_process()
    
    # Simulate some work
    time.sleep(np.random.uniform(0, 1))    
    
    # Print information about which process is handling this element
    print(file_to_process, "running on CPU", process.name[-1])
    return file_to_process

# Get the number of CPUs available
num_cpus = multiprocessing.cpu_count()
print(f"System has {num_cpus} CPUs available")

# Example list of 20 elements to process
files_to_process = [f"File_{i}" for i in range(20)]
print(f"Processing {len(files_to_process)} files...")

# Create a process pool with maximum available CPUs
with multiprocessing.Pool(processes=num_cpus) as pool:
    # Map the worker function across all elements
    results = pool.map(worker_function, files_to_process)
    
    print("\nAll files processed!")
    print(f"Results: {results}")