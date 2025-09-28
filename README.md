
## Setup Environment

```bash
# SSH into the server:
ssh vector

# Start an interactive session with the Slurm job scheduler (A100 or T4):
srun --gres=gpu:1 --partition=a100 --qos=a100_mmai --time=08:00:00 --pty bash
# or
srun --gres=gpu:1 --partition=t4v2 --time=08:00:00 --mem=16G --pty bash

# In the interactive session, install the Python kernels and start a jupyter session:
source /projects/aieng/multimodal_bootcamp/envs/vlm_env/bin/activate
python -m IPython kernel install --user --name=vlm_env --display-name "VLM Env (GPU)"
cd ~/projects/chartqa_visual_rag
python -m jupyter notebook --ip $(hostname --fqdn) --port 8899

# From a local terminal, create an SSH tunnel to the Jupyter notebook (Lookup the port and gpu number from last step)
ssh vector -L 8899:gpu###:8899

# Follow the link given in the jupyter session log: http://127.0.0.1:8899/tree?token=xxx (edited) 
```