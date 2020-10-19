#!/usr/bin/env python3
#!coding=utf-8
 
#right code !
#function: 
#display the frame from another node.
 
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from std_msgs.msg import Header
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import os
from sys import platform
import argparse
#from more_interfaces.msg import Arr



class Abo(Node):
	
	#image_path = "/home/student/openpose/examples/image/1.jpeg"
	#cv_img = cv2.imread(image_path)
    
	def __init__(self):
		super().__init__('abo')
		global frame_id 
		frame_id = 0
		# OPENPOSE INITIALIZATION
		try:
			# Import Openpose (Windows/Ubuntu/OSX)
			dir_path = os.path.dirname(os.path.realpath(__file__))
			try:
				# Windows Import
				if platform == "win32":
					# Change these variables to point to the correct folder (Release/x64 etc.)
					sys.path.append(dir_path + '/../../python/openpose/Release');
					os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
					import pyopenpose as op
				else:
					# Change these variables to point to the correct folder (Release/x64 etc.)
					#sys.path.append('../../python');
					# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
					sys.path.append('/usr/local/python')
					from openpose import pyopenpose as op
			except ImportError as e:
				print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
				raise e

			# Flags
			parser = argparse.ArgumentParser()
			parser.add_argument("--image_path", default="/home/student/openpose/examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
			args = parser.parse_known_args()

			# Custom Params (refer to include/openpose/flags.hpp for more parameters)
			params = dict()
			params["model_folder"] = "/home/student/openpose/models/"
			params["face"] = True
			params["hand"] = True

			# Add others in path?
			for i in range(0, len(args[1])):
				curr_item = args[1][i]
				if i != len(args[1])-1: next_item = args[1][i+1]
				else: next_item = "1"
				if "--" in curr_item and "--" in next_item:
					key = curr_item.replace('-','')
					if key not in params:	params[key] = "1"
				elif "--" in curr_item and "--" not in next_item:
					key = curr_item.replace('-','')
					if key not in params:	params[key] = next_item

			# Starting OpenPose
			self.opWrapper = op.WrapperPython()
			self.opWrapper.configure(params)
			self.opWrapper.start()
			# Process Image
			self.datum = op.Datum()
			self.poseModel = op.PoseModel.BODY_25
		except Exception as e:
			print(e)
			sys.exit(-1)

		#subscriber and publisher initialization
		self.br = CvBridge()
		self.subscription = self.create_subscription(Image,'/camera/rgb/image_raw',self.callback,10)
		self.publisher1 = self.create_publisher(Image,'/pose',10)
		self.publisher2 = self.create_publisher(Float32MultiArray,'/keypoints',10)
		self.subscription   #prevent unused variable warning
		


	def callback(self,data):
		#input image
		cv_img = self.br.imgmsg_to_cv2(data, "bgr8")

		#image_path = "/home/student/openpose/examples/image/2.jpg"
		#imageToProcess = cv2.imread(image_path)
		self.datum.cvInputData = cv_img	#imageToProcess
		self.opWrapper.emplaceAndPop([self.datum])

		# array config
		keypoints = Float32MultiArray()
		#keypoints.layout.data_offset = 0
		#keypoints.layout.dim = [MultiArrayDimension(), MultiArrayDimension(), MultiArrayDimension()]
		#d = str(self.datum.poseKeypoints.flatten())
		#keypoints.layout.dim[0].size = k
		#keypoints.layout.dim[1].size = j
		#keypoints.layout.dim[2].size = i
		#keypoints.layout.dim[0].stride = i * j * k
		#keypoints.layout.dim[1].stride = j * k
		#keypoints.layout.dim[2].stride = k
		#keypoints.data = [0]*(i * j * k)
		#keypoints.arr = self.datum.poseKeypoints.flatten()
		try:
			[i,j,k] = np.array(self.datum.poseKeypoints).shape
		except ValueError:
			print('ERROR: No one detected!')
			sys.exit(-1)

		#save datum.poseKeypoints in a numpy array
		A = np.zeros(i*j*k)
		for x in range(i):
			for y in range(j):
				for z in range(k):
					A[x * k *j + y * k + z] = self.datum.poseKeypoints[x][y][z]
		#tuple!
		tup = tuple(A)
		keypoints.data = tup
		#print(keypoints)

		#print keypoints
		#print("Body keypoints: \n" + str(keypoints))
		#print("Face keypoints: \n" + str(self.datum.faceKeypoints))
		#print("Left hand keypoints: \n" + str(self.datum.handKeypoints[0]))
		#print("Right hand keypoints: \n" + str(self.datum.handKeypoints[1]))
		
		#cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", self.datum.cvOutputData)
		#cv2.waitKey(0)

		#save results
		time = Header()
		time.stamp = Node.get_clock(self).now().to_msg()
		cv2.imwrite('/home/student/pose/'+str(time)+'.jpg',self.datum.cvOutputData)
		#frame_id = frame_id + 1
	
		#publish results
		self.publisher1.publish(self.br.cv2_to_imgmsg(np.array(self.datum.cvOutputData), "bgr8"))
		#data2 = np.array(self.datum.poseKeypoints).flatten()
		#print("Body keypoints: \n" + str(keypoints))
		self.publisher2.publish(keypoints)
		

def main(args=None):
	rclpy.init(args=args)

	abo = Abo()
	rclpy.spin(abo)  #loop

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	abo.destroy_node()
	rclpy.shutdown()
 
if __name__ == '__main__':
	main()

