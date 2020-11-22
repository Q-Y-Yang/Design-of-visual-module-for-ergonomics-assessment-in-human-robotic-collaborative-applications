#!/usr/bin/env python
# coding=UTF-8

import rospy
import message_filters
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



def callback(side_data, front_data):
	br = CvBridge()
	side_img  = br.imgmsg_to_cv2(side_data, "bgr8")
	front_img = br.imgmsg_to_cv2(front_data, "bgr8")
 
 
	syn_img = np.concatenate((side_img, front_img), axis=1 )
 
	pub.publish(br.cv2_to_imgmsg(syn_img,"bgr8"))

def img_synchronizer():
	rospy.init_node('image_synchronizer', anonymous=True)

	global pub
	pub = rospy.Publisher('syn_img', Image, queue_size=10)

	sub_side  = message_filters.Subscriber("/logi_c922_1/image_raw", Image, queue_size=1, buff_size=110592*6 )
	sub_front = message_filters.Subscriber("/logi_c922_2/image_raw", Image, queue_size=1, buff_size=110592*6 )
 
	ts = message_filters.ApproximateTimeSynchronizer([sub_side, sub_front], 10, 0.1, allow_headerless = True)
	ts.registerCallback(callback)


 

 
     # spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__== '__main__':
     try:
         img_synchronizer()
     except rospy.ROSInterruptException:
         pass
