# Human-estimation-001_plus_tracker
Tracker is added into the original human-estimation-001 model. With this model, we can get one's poses across frames. Therefore, we can track velocties of all people in the frames as well as their poses for fall detection.
# Tunable parameters
```
veloctiy_threshold: a condition to determine whether one falls or not (float)
```
you can tune the above parameters by, for example:<br />
``` python3 main.py --velocity_threshold 150 ```
# Requirements
- To install necessary packages for fall detection model

     ```pip3 install -r requirements.txt```

# Usage
- To run the model:

    ```python3 main.py```
