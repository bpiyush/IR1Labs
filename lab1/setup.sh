# Usage: bash setup.sh
# Outcome: Creates a conda environment for the lab1 and installs the required packages

echo "::: Creating conda environment for lab1 using conda version $(conda --version)"
echo "::: This has been tested on Mac OS with I386 architecture."

conda create -y -n ir1-i386
conda activate ir1-i386
conda install python=3.6.5 --channel conda-forge
pip install -r ./requirements.txt
pip install jupyterlab ipdb tqdm

echo "::: Setup complete."