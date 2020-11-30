# Design-of-visual-module-e-for-ergonomics-assessment-in-human-robotic-collaborative-applications
Research Internship TUM

pose_ergonomic
--
* A ROS 2 package to subscribe synchronized images as input, detect 2D body and hands pose via openpose, performing ergonomic assessment and publish results.<br>

* Installing [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) firstly following its [prerequisites](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/prerequisites.md) and [installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/doc/installation) instructions.<br>
* Build in ROS 2 workspace: `colcon build --packages-select pose_ergonomic`<br>
Remember to source after building: `. install/setup.bash`<br>
Run: `ros2 run pose_ergonomic pose_ergonomic.py`<br>

* Three result publishers: <br>
1. '\pose': publish images with keypoints annotations.<br>
2. '\keypoints': publish keypoints coordinates.<br>
3. '\risk': publish risk levels from ergonomic assessment.<br>

* `ergonomic evaluation` module: performing ergonomic assessment according to Rapid Upper Limbs Assessment(RULA) criteria based on 2D body and hands keypoints obtained from openpose.<br>


syn_img_pub
--
* Subscribe two topics synchronously using messages filters in ROS Melodic.<br>
* Approximate time synchronously subscribe images(side view and front-top view) from two cameras, rotate and resize side view image to make the two synchronized images into one image for later processing by OpenPose.<br>
Run: `rosrun syn_img_pub syn_img_pub`<br>

ROS1 Bridge
--
* ROS1 Bridge is necessary to transmit synchronized images from ROS melodic to ROS eloquent.
* Following the "build the ROS1 Bridge" and "run the ROS1 Bridge" parts of instructions [here](https://industrial-training-master.readthedocs.io/en/melodic/_source/session7/ROS1-ROS2-bridge.html) to build and run your ROS1 Bridge.

