import time
import os
import nvidia_smi

Seconds = 5 
OutPath = "/home/charlie/Desktop/Practical/GPU_Measuring/Performance"
FileName = "_Performance_Log.csv"
TimeStamp = time.strftime("%Y%m%d_%H%M%S")

OutputFile = os.path.join(OutPath, f"{TimeStamp}{FileName}")
header = "Time, GPU, Memory Total (MiB), Memory Used (MiB)\n"

# Create the log file with a header
with open(OutputFile, 'w') as f:
    f.write(header)

def monitor_gpu(log_file=OutputFile, interval=Seconds):
    # Initialise NVML
    nvidia_smi.nvmlInit()

    with open(log_file, 'a') as f:
        try:
            while True:
                deviceCount = nvidia_smi.nvmlDeviceGetCount()
                
                for i in range(deviceCount):
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    gpu_name = nvidia_smi.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    CurrentTimeStamp = time.strftime("%H:%M:%S")
                    
                    log_entry = (f"{CurrentTimeStamp}, {gpu_name}, {info.total // (1024**2)},{info.used // (1024**2)}")
                    print(log_entry) 
                    f.write(log_entry + "\n")
                    f.flush()  # Ensure data is written immediately

                time.sleep(interval)  
        except KeyboardInterrupt:
            print("GPU Monitoring stopped.")
        finally:
            nvidia_smi.nvmlShutdown() 

if __name__ == "__main__":
    monitor_gpu()
