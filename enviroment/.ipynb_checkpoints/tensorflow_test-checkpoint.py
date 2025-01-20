import tensorflow as tf
import time

# Record start time
start_time = time.time()

# Create a tensor and perform some operation to test the GPU
tensor = tf.random.normal([10000, 10000])
result = tf.matmul(tensor, tensor)

# Record end time
end_time = time.time()

# Print the time taken
print(f"Time taken: {end_time - start_time} seconds")

