import os


coda_out_dir = "coda_out/"
training_log_path = "/training_log/log.txt"

os.makedirs(coda_out_dir, exist_ok=True)

loss_logging_path = coda_out_dir + "log/loss.log"
os.makedirs(coda_out_dir + "/log", exist_ok=True)

weight_saving_dir = coda_out_dir + "weight/"
os.makedirs(weight_saving_dir, exist_ok=True)
