import os
import time

# Configuration
epochs = 1
patience = 3
input_len = 50
output_len = 50
features = 'T'
target = 'Open'
loss = 'MSE'
data_file = 'USATECH.IDXUSD_Candlestick_1_M_BID_01.01.2022-01.01.2024.csv'
model_file = 'checkpoint_autoformer.pth'
data = 'nasdaq'
name = 'NASDAQ_TEST'
CUDA_DEBUG = False

# Utility function
def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

# Check environment
USING_COLAB = False
try:
    from google.colab import drive
    drive.mount('/content/drive')
    USING_COLAB = True
except ModuleNotFoundError:
    print("Google Colab environment not detected, running locally.")

# CUDA Debugging
if CUDA_DEBUG:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Record the start time
notebook_start_time = time.time()

# Command to run the script
command = (
    f"python -u ./run.py "
    f"--task_name long_term_forecast "
    f"--is_training 0 "
    f"--is_inferencing 1 "
    f"--root_path ./ "
    f"--data_path {data_file} "
    f"--inference_path {model_file} "
    f"--model_id {name}_{input_len}_{output_len} "
    f"--model Autoformer "
    f"--data {data} "
    f"--features {features} "
    f"--freq t "
    f"--train_epochs {epochs} "
    f"--patience {patience} "
    f"--inverse "
    f"--target {target} "
    f"--seq_len {input_len} "
    f"--label_len 50 "
    f"--pred_len {output_len} "
    f"--batch_size 1 "
    f"--loss {loss} "
    f"--embed fixed "
    f"--e_layers 2 "
    f"--d_layers 1 "
    f"--factor 3 "
    f"--enc_in 1 "
    f"--dec_in 1 "
    f"--c_out 1 "
    f"--des 'Exp' "
    f"--itr 1"
)

# Execute the command
os.system(command)

# Record the end time
notebook_end_time = time.time()

# Calculate the duration
notebook_duration = notebook_end_time - notebook_start_time
print(f"Notebook execution succeeded in {format_time(notebook_duration)}.")
