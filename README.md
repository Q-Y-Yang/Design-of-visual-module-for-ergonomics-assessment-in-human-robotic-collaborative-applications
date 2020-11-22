# Design-of-visual-module-e-for-ergonomics-assessment-in-human-robotic-collaborative-applications
Research Internship TUM

Bild
--
ROS 2 node to subscribe images as input, process via Openpose and publish results<br>
build: `colcon build --packages-select bild`<br>
remember to source after building: `. install/setup.bash`<br>
run: `ros2 run bild abo`<br>
<br>
Two result publishers: 1. publish images with keypoints annotations.<br>
2. publish keypoints coordinates.<br>

sub
--
A ROS 2 node to subscribe keypoints coordinates stored in arraysand save as a txt file in path "\home\student\pose".<br>
Source ROS 2 workspace and then run:<br>
`ros2 run sub sub`<br>

syn_img_pub
--
Subscribe two topics synchronously using messages filters.<br>
Approximate time synchronously subscribe images from two cameras, make the synchronized images into one image for later processing by OpenPose.<br>
`rosrun syn_img_pub syn_img_pub.py`<br>

