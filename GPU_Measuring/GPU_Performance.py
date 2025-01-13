import time
import subprocess
import os
import pandas

Seconds= 5
OutPath= "/home/charlie/Desktop/Practical/GPU_Measuring/Performance"
FileName= "_Performance_Log.csv"
TimeStamp = time.strftime("%Y%m%d_%H%M%S")

OutputFile= os.path.join(OutPath,f"{TimeStamp}{FileName}")
header="Time,Percent,Memory(MiB)\n"

log_file=OutputFile
with open(log_file, 'w') as f:
    f.write(header)

def monitor_gpu(log_file=OutputFile, interval=Seconds):
    with open(log_file, 'a') as f:
        try:
            while True:
                N_SMI_OutPut = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'])
                CurrentTimeStamp = time.strftime("%Y-%m-%d %H:%M:%S")
                latest_memory_usage = df.iloc[-1]["Memory(MiB)"]
                #converts returned byte string to normal string
                f.write(f"{CurrentTimeStamp}, {N_SMI_OutPut.decode('utf-8').strip()}\n")
                #Outputs data immediately
                f.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            pass  

if __name__ == "__main__":
    monitor_gpu()

percent, memory_mib = map(int, gpu_output.split(", "))
print(f"Current Memory Usage: {memory_mib} MiB")
