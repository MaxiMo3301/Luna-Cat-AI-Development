# Human-estimation-001_multi_device
Run human-estimation-001 on multiple devices per host for 2d multi-persons pose detection and detect whether there is person fallen. So, you can connect multiples OAK cameras to a host for fall detection.
# Requirements (opencv-python, depthai, blobconverter)
To install required packages:

```pip3 install -r requirements.txt```
# RUN
To run the model in defualt mode:

``` python3 main.py ```

To run the model with skeleton image only (no rgb frame):

```python3 main.py -b ```

You can ajust size of the internal camera by:

``` python3 main.py --width 900 --height 900 ```
