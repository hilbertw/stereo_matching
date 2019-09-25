Stereo Matching
==================
The repository shows an example for stereo matching I've been learning recently. Support both cpu and gpu working.

2019-09-05: Support sky mask, using a modified version of [sky-detector](https://github.com/MaybeShewill-CV/sky-detector).

2019-09-25: Add LR-check.

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

![demo_disp.png](https://github.com/hunterlew/stereo_matching/blob/develop/example/demo_disp.png)
![demo_mapping.png](https://github.com/hunterlew/stereo_matching/blob/develop/example/demo_mapping.png)
