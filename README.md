# Design-of-visual-module-e-for-ergonomics-assessment-in-human-robotic-collaborative-applications
Research Internship at Technical University of Munich (TUM)

pose_ergonomic
--
* A ROS 2 package to subscribe synchronized images as input, detect 2D body and hands pose via openpose, performing ergonomic assessment and publish results.<br>

* Installing [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) firstly following its [prerequisites](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/prerequisites.md) and [installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/doc/installation) instructions.<br>
* Build in ROS 2 workspace: `colcon build --packages-select pose_ergonomic`<br>
Remember to source after building: `. install/setup.bash`<br>
Run: `ros2 run pose_ergonomic pose_ergonomic.py`<br>

* Input options: <br>
`--save_result`: True: save results; False: do not save.<br>
`--result_path`: a path to store result.<br>
`--single_view`: pose estimation from single view or two views to be further ergonomically evaluated.<br>
`--rot90`: clockwise rotate 90 degrees.<br>
`--camera_topic`: the camera topic to subscribe as input.<br>
`--ergonomic`: 1: perform RULA; 2: perform NERPA.<br>

* Output: <br>
1. `/pose`: publish images with keypoints estimation results.<br>
2. `/angles`: publish joint angular values.<br>
3. `/risk`: publish risk levels from ergonomic assessment.<br>

* `ergonomic evaluation` module: performing ergonomic assessment according to Rapid Upper Limbs Assessment(RULA) criteria based on 2D body and hands keypoints obtained from openpose.<br>
`ergonomic_nerpa` module: performing ergonomic assessment according to Novel Ergonomic Postural Assessment Method (NERPA).<br>


syn_img_pub
--
* Subscribe two topics synchronously using messages filters in ROS Melodic.<br>
* Approximate time synchronously subscribe images(side view and front-top view) from two cameras, rotate and resize side view image to make the two synchronized images into one image for later processing by OpenPose.<br>
* Output: `/syn_img`: synchronized and integrated image.<br>
*  Run: `rosrun syn_img_pub syn_img_pub`<br>

mocap_ergonomic
--
* A ROS 2 package to subscribe synchronized images as input, detect 3D body and hands pose via FrankMocap, performing ergonomic assessment and publish results.<br>

* Installing [FrankMocap](https://github.com/facebookresearch/frankmocap) firstly following its [installation](https://github.com/facebookresearch/frankmocap/blob/master/docs/INSTALL.md) instructions.<br>
* Build in ROS 2 workspace: `colcon build --packages-select mocap_ergonomic`<br>
Remember to source after building: `. install/setup.bash`<br>
Run: `ros2 run mocap_ergonomic mocaphttps://github.com/facebookresearch/frankmocap_ergonomic.py`<br>

syn_img_frank
--
* Subscribe two topics synchronously using messages filters in ROS Melodic.<br>
* Approximate time synchronously subscribe images(side view and front-top view) from two cameras, rotate and resize side view image. Finally, publish the synchronized images from different views sequentially for later processing by FrankMocap.<br>
* Output: `/syn_img`: synchronized and integrated image.<br>
* Run: `rosrun syn_img_frank syn_img_frank`<br>

* Input:<br>
Subscribing `/side_img` and `/front_img` processed by syn_img_frank.<br>
* Output:<br>
1. `/pose`: publish images with keypoints estimation results.<br>
2. `/angles`: publish joint angular values.<br>
3. `/risk`: publish risk levels from ergonomic assessment.<br>

* `eva3d` module: performing ergonomic assessment according to Rapid Upper Limbs Assessment(RULA) criteria based on 3D body and hands keypoints obtained from FrankMocap.<br>

ROS1 Bridge
--
* ROS1 Bridge is necessary to transmit synchronized images from ROS melodic to ROS eloquent.
* Following the "build the ROS1 Bridge" and "run the ROS1 Bridge" parts of instructions [here](https://industrial-training-master.readthedocs.io/en/melodic/_source/session7/ROS1-ROS2-bridge.html) to build and run your ROS1 Bridge.

Note
--
You may have problems when using modules from OpenPose and FrankMocap in these packages. Pay attention to the path.<br>
The installation paths of OpenPose and FrankMocap in this research internship are `/home/student/openpose` and `/home/studnet/frankmocap`.
