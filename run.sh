mkdir -p ./.carla
mkdir -p ./.carla/checkpoints
mkdir -p ./.carla/carla
wget -N http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.5.tar.gz
tar xzf ./CARLA_0.9.5.tar.gz -C ./.carla/carla
tar xzf ./.carla/checkpoints/exp40.tar.gz -C ./.carla/checkpoints/
