import subprocess
import os

# Step 1: Read the contents of neighbor.txt
with open('neighbor.txt', 'r') as file:
    lines = file.readlines()

# Step 2 & 3: Parse each line to extract the weights and construct the command
for line in lines:
    weights = line.strip().split(', ')
    if len(weights) == 3:  # Ensure there are exactly three weights
        # Use weights to modify model_dir and ckpt_dir
        model_dir = f"/tmp/pcc_saved_models/model_{weights[0]}_{weights[1]}_{weights[2]}/"
        ckpt_dir = f"./data_{weights[0]}_{weights[1]}_{weights[2]}/pcc_model"
        
        command = f"python stable_solve.py --weights {weights[0]} {weights[1]} {weights[2]} --model_dir {model_dir} --ckpt_dir {ckpt_dir}"
        
        print(command)
        print(f"cp ./data_{weights[0]}_{weights[1]}_{weights[2]}/pcc_model.ckpt.* .")
        # # Step 4: Execute the command
        # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # stdout, stderr = process.communicate()
        
        # # Optional: Print the output or handle errors
        # if process.returncode == 0:
        #     print(f"Command executed successfully: {command}")
        #     print(stdout.decode())
        # else:
        #     print(f"Error executing command: {command}")
        #     print(stderr.decode())