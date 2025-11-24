docker rm -f foundationpose
DIR=$(pwd)/third_party/FoundationPose

# Starts foundation pose docker container in detatched state (no terminal)
xhost +  && docker run -d --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE shingarey/foundationpose_custom_cuda121:latest

# Run build_all as per FoundationPose README
docker exec foundationpose bash -c "cd $DIR && bash ../../keyinsertion/foundationpose/build_all.sh"
