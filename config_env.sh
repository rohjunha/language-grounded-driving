conda update -y -n base conda
conda create -y -n carla_env python=3.7
conda activate carla_env
pip install --upgrade pip
conda install -y -n carla_env -c conda-forge ffmpeg
conda install -y -n carla_env tensorflow-gpu
conda install -y -n carla_env pytorch torchvision cudatoolkit=10.0 -c pytorch
