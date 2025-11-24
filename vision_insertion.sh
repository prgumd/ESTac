set -e

trial=19

# Locks
# lock_type="pin_tumbler"
# lock_diameter=0.150
# key_offset_z=0.132
# key_offset_y=0.001
# key_offset_x=0.00
# penetration_depth=0.015

# lock_type="tubular"
# lock_diameter=0.150
# key_offset_z=0.125
# key_offset_y=0.002
# key_offset_x=-0.002
# penetration_depth=0.005

# lock_type="disc_detainer"
# lock_diameter=0.150
# key_offset_z=0.133
# key_offset_y=0.0013
# key_offset_x=-0.003
# penetration_depth=0.018

# lock_type="dimpled"
# lock_diameter=0.150
# key_offset_z=0.137
# key_offset_y=0.0
# key_offset_x=0.0
# penetration_depth=0.018

# lock_type="00340"
# lock_diameter=0.073
# key_offset_z=0.120 #0.11 #0.115 #0.1125
# key_offset_y=0.012 #0.01 #0.0143
# key_offset_x=0.001
# penetration_depth=0.020

# lock_type="00320"
# lock_diameter=0.075
# key_offset_z=0.162
# key_offset_y=0.003
# key_offset_x=-0.005
# penetration_depth=0.013

# lock_type="00346"
# lock_diameter=0.07
# key_offset_z=0.144
# key_offset_y=0.004
# key_offset_x=0.000
# penetration_depth=0.005

# lock_type="00015"
# lock_diameter=0.03
# key_offset_z=0.141
# key_offset_y=0.002
# key_offset_x=-0.003
# penetration_depth=0.023

lock_type="00296"
lock_diameter=0.055
key_offset_z=0.130
key_offset_y=0.017
key_offset_x=0.006
penetration_depth=0.008

# Set the initial pose
# Estimate the masks with SAM2 (interactive)
python keyinsertion/vision_accuracy.py \
--key_offset_z ${key_offset_z} \
--key_offset_y ${key_offset_y} \
--key_offset_x ${key_offset_x} \
--lock_type ${lock_type}_insertion_${trial} \
--lock_diameter ${lock_diameter} \
--N_sample 100 \
--seed 1 \
--i_sample ${trial} \
--collect \
--mask

# Estimate the pose of the lock
# Check if foundationpose docker container is running. If not, start it.
if [ "$(docker ps -q -f name=foundationpose)" ]; then
    echo "Container foundationpose is running"
else
    echo "Container foundationpose is not running, starting it now..."
    bash keyinsertion/foundationpose/run_container.sh
fi

py_file="keyinsertion/foundationpose/estimate_pose.py"
mesh_file="keyinsertion/meshes/${lock_type}/mesh_reconstructed.obj"
data_dir="keyinsertion/data/${lock_type}_insertion_${trial}/"
docker exec foundationpose bash -c "cd $(pwd) && python ${py_file} --mesh ${mesh_file} --test_scene_dir ${data_dir} --wait"

read -p "Press any key to move the arm or 'q' to exit: " input
if [[ $input == "q" ]]; then
    exit 0
fi

# Move the arm to the lock opening using the estimated lock pose
python keyinsertion/vision_accuracy.py \
--key_offset_z ${key_offset_z} \
--key_offset_y ${key_offset_y} \
--key_offset_x ${key_offset_x} \
--lock_type ${lock_type}_insertion_${trial} \
--move

read -p "Press any key to start insertion or 'q' to exit: " input
if [[ $input == "q" ]]; then
    exit 0
fi

python keyinsertion/key_insertion.py --penetration_depth ${penetration_depth} \
--key_offset_z ${key_offset_z} \
--key_offset_y ${key_offset_y} \
--key_offset_x ${key_offset_x} \
--output_name ${lock_type}_insertion_${trial}
