mkdir -p ./.carla
mkdir -p ./.carla/checkpoints
mkdir -p ./.carla/carla
wget -N http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.5.tar.gz
tar xzf ./CARLA_0.9.5.tar.gz -C ./.carla/carla
wget -N http://54.201.45.51:5000/exp40/exp40.tar.gz
tar xzf ./exp40.tar.gz -C ./.carla/checkpoints/
