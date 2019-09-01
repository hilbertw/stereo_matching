Stereo Matching
==================
This repository is an example for stereo matching I've been learning recently. Support both cpu and gpu working.

2019-09-05: Support sky mask, using a modified version of [sky-detector](https://github.com/MaybeShewill-CV/sky-detector).

Pre-requisites
--------------
ROS

CUDA

OpenCV

Datasets
--------
The code is tested on [Kitti dataset](http://www.cvlibs.net/datasets/kitti/eval_stereo.php).

Result
-----
Below shows the disparity and coresponding pointcloud mapping visualized using rviz.

![disparity_0.jpg](https://github.com/hunterlew/stereo_matching/blob/develop/example/disp_0.png)
![mapping_0.jpg](https://github.com/hunterlew/stereo_matching/blob/develop/example/mapping_0.png)
