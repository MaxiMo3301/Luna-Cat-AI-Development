# Multi-person-depthai-blazepose
I attempt to create a multi-persons pose detection model using blazepose on depthai cameras. However, the estimated body poses by this model are deformed and cannot be used for fall detection. 
# Usage
- To install necessary packages for fall detection model

     ```pip3 install -r requirements.txt```
  
- To use default internal color camera as input with the model "full" for detecting fall:

    ```python3 demo.py -xyz --multi_detection```
