# Object tracker + Blazepose
A mobile net person tracker is added to track all people on a frame ,and pose detection is performed for each person on the frame    
# Issue
Seemingly, there is a problem in pre_landmark_imanip_node, where the frame of the person is cropped and is resized for the input of the blazepose model. 
Under some conditions, error 'ResourceLocker' '358' will be triggered. 
