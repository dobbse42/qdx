#!/bin/bash
#SBATCH --job-name=qdx_test
#SBATCH --account=project_2015248
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1,nvme:1

#SBATCH --output=output/%x.%j.out

# module load python-data
module purge
# module load gcc/13.2.0
# module load cuda/12.6.0
# module load cudnn
module load jax/0.4.23

#python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install -q -r requirements.txt
# pip install --upgrade "jax[cuda12-local]"

# python3 -m pip list
# nvcc --version

# python -c "import jax; print(f'Jax backend: {jax.default_backend()}')"

python main.py
