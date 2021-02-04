#!/usr/bin/env python
# coding=UTF-8

import rospy
import message_filters
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import argparse



def callback(side_data, front_data):
	br = CvBridge()
	side_img_raw  = br.imgmsg_to_cv2(side_data, "bgr8")
	front_img = br.imgmsg_to_cv2(front_data, "bgr8")

	side_imgcut = side_img_raw[:,160:,:]  #:480frontknee(and tracker) or 160: front side: to cut image 
	h, w = side_imgcut.shape[:2]
	center = (w // 2, h // 2)
	Rotate = cv2.getRotationMatrix2D(center, -90, 1) #90 frontknee(and tracker and  non object) or -90 front side,
	side_imgr = cv2.warpAffine(side_imgcut, Rotate, (w, h))
	side_img = side_imgr[:,230:]	#for side view to avoid window
	#side_img = side_imgr[:,:350]	#for tracker to avoid something near door(detected as a human)
	
	#syn_img = np.concatenate((side_img, front_img), axis=1 )

	#cv2.imwrite('/home/student/frontknee_dataset/'+str(side_data.header.seq)+'.jpg',syn_img)
	side_img = br.cv2_to_imgmsg(side_img,"bgr8")
	side_img.header = side_data.header
	front_img = br.cv2_to_imgmsg(front_img,"bgr8")
	front_img.header = front_data.header
	side_pub.publish(side_img)
	front_pub.publish(front_img)

def img_synchronizer():
	rospy.init_node('image_synchronizer', anonymous=True)

	global side_pub
	global front_pub
	side_pub = rospy.Publisher('side_img', Image, queue_size=10)
	front_pub = rospy.Publisher('front_img', Image, queue_size=10)

	sub_side  = message_filters.Subscriber("/logi_c922_1/image_rect_color", Image, queue_size=1, buff_size=110592*6 )
	sub_front = message_filters.Subscriber("/logi_c922_2/image_rect_color", Image, queue_size=1, buff_size=110592*6 )
 
	ts = message_filters.ApproximateTimeSynchronizer([sub_side, sub_front], 10, 0.1, allow_headerless = True)
	ts.registerCallback(callback)
     # spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__== '__main__':
     try:
         img_synchronizer()
     except rospy.ROSInterruptException:
         pass
