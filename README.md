# Design-of-visual-module-e-for-ergonomics-assessment-in-human-robotic-collaborative-applications
Research Internship TUM

pose_ergonomic
--
A ROS 2 package to subscribe synchronized images as input, detect 2D body and hands pose via openpose, performing ergonomic assessment and publish results.<br>
<br>
Installing [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) firstly following its [prerequisites](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/prerequisites.md) and [installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/doc/installation) instructions.<br><br>
build in ROS 2 workspace: `colcon build --packages-select pose_ergonomic`<br>
remember to source after building: `. install/setup.bash`<br>
run: `ros2 run bild pose_ergonomic.py`<br>
<br>
Two result publishers: <br>
1. publish images with keypoints annotations.<br>
2. publish keypoints coordinates.<br><br>

`ergonomic evaluation` module: performing ergonomic assessment according to Rapid Upper Limbs Assessment(RULA) criteria based on 2D body and hands keypoints obtained from openpose.<br>


syn_img_pub
--
Subscribe two topics synchronously using messages filters.<br>
Approximate time synchronously subscribe images from two cameras, make the synchronized images into one image for later processing by OpenPose.<br>
`rosrun syn_img_pub syn_img_pub.py`<br>

