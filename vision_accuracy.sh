set -e

# Locks
# lock_type="pin_tumbler"
# lock_diameter=0.150
# key_offset_z=0.137
# key_offset_y=0.001
# key_offset_x=0.0

# lock_type="tubular"
# lock_diameter=0.150
# key_offset_z=0.124
# key_offset_y=0.002
# key_offset_x=0.0

# lock_type="disc_detainer"
# lock_diameter=0.150
# key_offset_z=0.131
# key_offset_y=0.0013
# key_offset_x=0.0

# lock_type="dimpled"
# lock_diameter=0.150
# key_offset_z=0.130
# key_offset_y=0.001
# key_offset_x=0.0

# AutoMate objects in order of mounting on cabinet
# lock_type="00340"
# lock_diameter=0.073
# key_offset_z=0.11 #0.115 #0.1125
# key_offset_y=0.015 #0.01 #0.0143
# key_offset_x=-0.005

# lock_type="00320"
# lock_diameter=0.075
# key_offset_z=0.155
# key_offset_y=-0.001
# key_offset_x=0.0

lock_type="00346"
lock_diameter=0.07
key_offset_z=0.137
key_offset_y=0.001
key_offset_x=0.000

# lock_type="00015"
# lock_diameter=0.03
# key_offset_z=0.135
# key_offset_y=0.003
# key_offset_x=-0.002

# lock_type="00296"
# lock_diameter=0.055
# key_offset_z=0.120
# key_offset_y=0.015
# key_offset_x=0.0

if [ "$1" == "collect" ]; then
    # Collect the data
    python keyinsertion/vision_accuracy.py \
    --key_offset_z ${key_offset_z} \
    --key_offset_y ${key_offset_y} \
    --key_offset_x ${key_offset_x} \
    --lock_type ${lock_type} \
    --lock_diameter ${lock_diameter} \
    --N_sample 100 \
    --seed 1 \
    --collect
fi

if [ "$1" == "mask" ]; then
    # Estimate the masks with SAM2 (interactive)
    python keyinsertion/vision_accuracy.py \
    --key_offset_z ${key_offset_z} \
    --key_offset_y ${key_offset_y} \
    --key_offset_x ${key_offset_x} \
    --lock_type ${lock_type} \
    --lock_diameter ${lock_diameter} \
    --mask
fi

if [ "$1" == "pose" ]; then
    # Check if foundationpose docker container is running. If not, start it.
    if [ "$(docker ps -q -f name=foundationpose)" ]; then
        echo "Container foundationpose is running"
    else
        echo "Container foundationpose is not running, starting it now..."
        bash keyinsertion/foundationpose/run_container.sh
    fi

    py_file="keyinsertion/foundationpose/estimate_pose.py"
    mesh_file="keyinsertion/meshes/${lock_type}/mesh_reconstructed.obj"
    data_dir="keyinsertion/data/${lock_type}/"
    docker exec foundationpose bash -c "cd $(pwd) && python ${py_file} --mesh ${mesh_file} --test_scene_dir ${data_dir}"
fi

if [ "$1" == "accuracy" ]; then
    # Calculate basic accuracy statistics and plot results
    python keyinsertion/vision_accuracy.py \
    --key_offset_z ${key_offset_z} \
    --key_offset_y ${key_offset_y} \
    --key_offset_x ${key_offset_x} \
    --lock_type ${lock_type} \
    --lock_diameter ${lock_diameter} \
    --accuracy
fi
