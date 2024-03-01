# Depthai blazepose for single person
Blazepose can estimate one's pose in a 3-dimensional space when using depthai camereas that have stereo cameras.
This code is bulit on top of [depthai blazepose](https://github.com/geaxgx/depthai_blazepose) which is developed by geaxgx. So, for detail of balzepose, you can refer to that repository.
Since blazepose used here is single pose detection model, the fall detection model is also for single person only.

Compared with the original depthai blazepose, fall_detection.py is added and demo.py is modified in this repository.
# Tunable parameters for fall detection
These paramenters can be tuned for better fall detection performance.
```
veloctiy_threshold: a condition to determine whether one falls or not (float)
f: fps of the internal camera (an integer)
detection_mode: image or mixed or world (mixed requires stereo cams but world and image does not)
lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1)
pd_score_thresh : confidence score to determine whether a detection is reliable (a float between 0 and 1)
```
you can tune the above parameters by, for example:<br />
``` python3 demo.py -e -xyz -f 17 --velocity_threshold 0.20 --detection_mode mixed --lm_score_thresh 0.63 --pd_score_thresh 0.5 ```
# Requirements
- To install necessary packages for fall detection model

     ```pip3 install -r requirements.txt```

  Notice that ```open3d``` is optional. If you want 3d visualization, you can install it and uncomment line 5 in BlazeposeRenderer.py. 

# Usage
- To use default internal color camera as input with the model "full" for detecting fall:

    ```python3 demo.py -e -xyz```
- You can show the detected skeleton only by adding -b argument:
  
   ```python3 demo.py -e -xyz -b```

- You can show the statistics by adding --show_stat argument:
  
   ```python3 demo.py -e -xyz --show_stat```

Remark: Do not enter ```python3 demo.py -e```, ```Fatal error. Please report to developers. Log: 'ResourceLocker' '358' ``` may occur. 
