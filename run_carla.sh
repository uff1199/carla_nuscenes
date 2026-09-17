CARLA="/Experiments/Carla/CARLA_d71f3947b-dirty/"

while true; do

    pushd $CARLA
    ./CarlaUE4.sh -RenderOffScreen -nosound

    sleep 30

done