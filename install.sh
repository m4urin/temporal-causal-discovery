# Create the environment 'temporal-causal-discovery' with Python 3.10.13
conda create -n temporal-causal-discovery python=3.10.13

# Activate the environment
conda activate temporal-causal-discovery

# Install the libraries within the activated environment
conda install numpy pandas scipy scikit-learn matplotlib hyperopt tqdm mlflow pytorch torchvision torchaudio pytorch-cuda=12.1 -c anaconda -c conda-forge -c pytorch -c nvidia -y
