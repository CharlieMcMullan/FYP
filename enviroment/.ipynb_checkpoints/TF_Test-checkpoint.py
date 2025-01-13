import tensorflow as tf

def check_tensorflow():
    try:
        # Check TensorFlow version
        print("TensorFlow version:", tf.__version__)
        
        # Perform a simple computation
        print("Running a simple computation to test TensorFlow...")
        a = tf.constant(10)
        b = tf.constant(20)
        result = tf.add(a, b)
        print(f"Result of computation (10 + 20): {result.numpy()}")
        
        # Check GPU availability
        print("Checking for GPU availability...")
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"GPU is available: {gpu_devices}")
        else:
            print("No GPU detected. Using CPU.")
            
        print("TensorFlow is working correctly.")
    except Exception as e:
        print("An error occurred while testing TensorFlow:")
        print(e)

if __name__ == "__main__":
    check_tensorflow()

