import numpy as np
import time

def process_chunk(chunk):
    """Function to process a chunk of data."""
    return [x * x * np.exp(x) for x in chunk]

def divide_and_process(data, n_chunks):
    """Divide data into chunks and process them sequentially."""
    chunk_size = len(data) // n_chunks
    results = []
    for i in range(n_chunks):
        start_index = i * chunk_size
        if i == n_chunks - 1:  # Include the remainder in the last chunk
            end_index = len(data)
        else:
            end_index = start_index + chunk_size
        chunk = data[start_index:end_index]
        results.extend(process_chunk(chunk))
    return results

# Generate a large dataset
data = np.random.randint(1, 100, size=10000000)

# Measure time for sequential processing
start_time_seq = time.time()
results_seq = divide_and_process(data, 10)
end_time_seq = time.time()

print("Sequential processing time:", end_time_seq - start_time_seq, "seconds")


from concurrent.futures import ProcessPoolExecutor, as_completed

def divide_and_process_concurrent(data, n_chunks):
    """Divide data into chunks and process them in parallel using ProcessPoolExecutor from concurrent.futures."""
    chunk_size = len(data) // n_chunks
    futures = []
    results = [None] * n_chunks  # Pre-allocate a list to hold the results
    with ProcessPoolExecutor(max_workers=n_chunks) as executor:
        for i in range(n_chunks):
            start_index = i * chunk_size
            if i == n_chunks - 1:  # Include the remainder in the last chunk
                end_index = len(data)
            else:
                end_index = start_index + chunk_size
            chunk = data[start_index:end_index]
            # Submit the chunk to be processed in a separate process
            futures.append((executor.submit(process_chunk, chunk), i))
        
        for future, position in futures:
            results[position] = future.result()
    
    # Flatten the list of lists to get a single list of results
    return [item for sublist in results for item in sublist]

# Generate a large dataset
data = np.random.randint(1, 100, size=1000000)

# Measure time for parallel processing with ProcessPoolExecutor
start_time_con = time.time()
results_con = divide_and_process_concurrent(data, 10)
end_time_con = time.time()

print("Parallel processing time with ProcessPoolExecutor:", end_time_con - start_time_con, "seconds")


