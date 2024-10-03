# Extremum Seeking Controlled Wiggling for Tactile Insertion

[Webpage](https://prg.cs.umd.edu/ESTac), [arXiv]()

[![Key Insertion Video](https://img.youtube.com/vi/wudA5iLC-cI/0.jpg)](https://www.youtube.com/watch?v=wudA5iLC-cI)

## Contents
* `key_insertion.py`: the algorithm that inserts the key
* `make_figures.py`: code to make the plot shown in the paper
* `show_output.py`: code to make a detailed plot for controller tuning

## Data
The recorded data for all 360 trials is available [here]()

## Dependencies
* numpy, scipy, matplotlib
* OpenCV
* [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/index.html)
* for plotting: [scienceplots](https://github.com/garrettj403/SciencePlots)
* `vme_research`: A minimal version of an internal library used by the PRG group for robotics research. The version and relevant components used with the paper is distributed in the vme_research folder with its own license. To install run: `pip install -e ./vme_research` from the root of this repository.
