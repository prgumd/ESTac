# Extremum Seeking Controlled Wiggling for Tactile Insertion

[Webpage](https://prg.cs.umd.edu/ESTac), [arXiv](https://arxiv.org/abs/2410.02595)

[![Key Insertion Video](https://img.youtube.com/vi/jRHIyMdj2NE/0.jpg)](https://www.youtube.com/watch?v=jRHIyMdj2NE)

## Contents
* `key_insertion.py`: the extremum seeking algorithm
* `key_insertion_cma.py`: the CMA-ES baseline
* `vision_accuracy.py/sh`: scripts for computing lock pose from vision and plotting the accuracy
* `vision_insertion.sh`: conducts a complete vision initialized insertion from start to finish
* `realsense.py`: hardware interfacing with the Realsense D435i camera used for vision based initialization
* `foundationpose/`: scripts for running the FoundationPose docker container that was used estimate socket pose

## Data
The recorded data for all 360 trials of our method is available [here](https://drive.google.com/file/d/1mgqrAPzIa9R8Qd21aqWgDSGF-uZGYUZY/view?usp=sharing).

Data collected during vision based insertions is available [here](https://drive.google.com/file/d/1cHmrzy39pqDfPGm0_VAB-LcK2YKBbalr/view?usp=sharing).

Data collected using the CMA-ES method is available [here](https://drive.google.com/file/d/1Nr2TY66RBKI87YAz_Ak_NN7HyXUEHQXB/view?usp=sharing).

The AutoMate objects with recreated chamfers are available [here](https://drive.google.com/file/d/1mNFH25jzUWby8rGUg4dv9_jTSQz3UUwl/view?usp=sharing). Note that object 00015's plug was chamfered correctly in AutoMate's released files and so is not included here.

The 3D meshes used with FoundationPose are availabled [here](https://drive.google.com/file/d/1S7INtCZdQVxamzPmGh2UDeb2Uf8dQz2q/view?usp=sharing).

## Dependencies
* `vme_research`: A minimal version of an internal library used by the group for robotics research. To install run: `pip install -e ./vme_research` from the root of this repository.
* numpy, scipy, matplotlib, jax, open3d
* OpenCV
* [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/index.html)
* [`SAM2`](https://github.com/facebookresearch/sam2)
* [FoundationPose](https://github.com/NVlabs/FoundationPose)
* [pycma](https://github.com/CMA-ES/pycma)

