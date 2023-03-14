
import time
import subprocess
import os

def stream_process(process):
    go = process.poll() is None
    for line in process.stdout:
        print(line.decode("utf-8"))
    return go

def run_cmd_with_output(cmd:str):
	process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	while stream_process(process):
		time.sleep(0.1)
	time.sleep(10)

user_path = os.path.abspath("/mnt/c/Users/uiyunkim/Documents/GitHub/remote-work/")

run_cmd_with_output(f"ENABLE_GPU={True} \
		              USER_PATH={user_path} \
                      CONTAINER_IMAGE={'jupyter-server'} \
                      TRAINING_IMAGE={'jupyter:minimal-notebook'} \
                      docker-compose build")

run_cmd_with_output(f"ENABLE_GPU={True} \
		              USER_PATH={user_path} \
                      CONTAINER_IMAGE={'jupyter-server'} \
                      TRAINING_IMAGE={'jupyter:minimal-notebook'} \
		              docker-compose -f docker-compose.yaml up -d")