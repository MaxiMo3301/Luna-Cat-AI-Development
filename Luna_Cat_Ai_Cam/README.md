# Luna-Cat-AI-cam
AI solutions for fall detection using computer vision. Depthai camereas (eg. OAK-D, OAK-1, ...) are needed for running the codes. The basic idea to detect if someone falls is to analyze one's pose. Therefore, the fall detection models consists of pose estimation phase and pose analyzation phase.

This repository contains three fall detection models and some experiments. 

- fall detection models
    - blazepose
    - human-estimation-001-multi-persons
    - human-estimation-001 + person tracker
- experiments

    This directory contains experiments on blazepose model.


Notes: Python 3.9 was used in the development
# Reference
[depthai_blazepose](https://github.com/geaxgx/depthai_blazepose/tree/main) by geaxgx <br/>
[gen2_human_pose](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose)
