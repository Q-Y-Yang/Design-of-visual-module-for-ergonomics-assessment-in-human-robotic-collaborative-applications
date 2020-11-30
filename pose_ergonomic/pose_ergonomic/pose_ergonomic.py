#!/usr/bin/env python3
#!coding=utf-8
 

#function: 
#Subscribe synchronized frames from different cameras
#Detect body and hands keypoints by OpenPose python API
#Performing Rapid Upper Limbs Assessment(RULA) to evaluate ergonomics
#inputs: two synchronized image messages with header including timestamp captured by two cameras from side view and front-top view
#outputs: Image with keypoints annotation,  keypoints coordinates, risk level of ergonomic assessment 
 
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int8
from std_msgs.msg import String
from std_msgs.msg import Header
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import os
from sys import platform
import argparse
import message_filters
from . import ergonomic_evaluation

camera_topic = "syn_img"

class pose_ergonomic(Node):

	#image_path = "/home/student/openpose/examples/image/1.jpeg"
	#cv_img = cv2.imread(image_path)
    
	def __init__(self):
		super().__init__('pose_ergonomic')
		
		# OPpenPose Initialization
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
		#input subscriber
		self.br = CvBridge()
		self.subscription_synimg = self.create_subscription(Image, camera_topic, self.callback,10)
		
		#output publisher
		self.publisher_pose = self.create_publisher(Image,'/pose',10)	#images with keypoints annotation
		self.publisher_keypoints = self.create_publisher(Float32MultiArray,'/keypoints',10)	#keypoints coordinates
		self.publisher_risk = self.create_publisher(Int8,'/risk',10)	#risk level
		#self.subscription   #prevent unused variable warning
		self.current_timestamp = 1     #timestamp init
		self.main_angles = np.empty((1,4),dtype = float)    #to store some calculated angle results

       #average calculation
		self.old_average = np.empty((1,4),dtype = float)     
		self._average = np.empty((1,4),dtype = float) 

		self.frame_No = 0      #frame number init in each one minute
		self.frame_id = 0	#count frames

       #variance calculation
		self.S = np.zeros((1,4),dtype = float) 
		self.variance = np.zeros((1,4),dtype = float) 

	def callback(self,data):
		#input image
	
		try:
			cv_img = self.br.imgmsg_to_cv2(data, "bgr8")
			self.frame_No = self.frame_No + 1
			self.frame_id = self.frame_id + 1
			self.old_average = self._average

			self.datum.cvInputData = cv_img	#imageToProcess
		except CvBridgeError as e:
			print(e)
		self.opWrapper.emplaceAndPop([self.datum])		
		
		# array msg init
		keypoints_whole = Float64MultiArray()
		
	
		#check if detect a person and a hand, if no, exist
		try:
			[i,j,k] = np.array(self.datum.poseKeypoints).shape  #i persons, j keypoints, k=3
		except ValueError:
			print('ERROR: No one detected!')	
			sys.exit(-1)
		try:
			[l,m,n,r] = np.array(self.datum.handKeypoints).shape	#l hands, m persons, n keypoints, r=3
		except ValueError:
			print('ERROR: Hand not detected!')
			sys.exit(-1)
		
		#check if there is two persons detected
		if i == 2:
			#save datum.poseKeypoints in a numpy array
			
			body = np.array(self.datum.poseKeypoints)
			hands = np.array(self.datum.handKeypoints)

			keypoints_side = np.concatenate((body[0],hands[0][0],hands[0][1]))
			keypoints_front = np.concatenate((body[1],hands[1][0],hands[1][1]))
	

			#tuple! to publish array msgs
			whole_body = np.concatenate((keypoints_side, keypoints_front))
			tup_wb = tuple(whole_body)
			keypoints_whole.data = tup_wb
			
			#to ergonomic evalution
			#whole_body = np.array(whole_body).reshape(i * 25 + m * l * 21, 3)
			
			#for muscle strain in one minute evaluation
			if data.header.stamp.sec  > self.current_timestamp + 60:  #check one minute
				self.current_timestamp = data.header.stamp.sec
		 
		        #reset after one minute
				self.frame_No = 0
				self.variance = self.old_average = self._average = self.S = np.zeros((1,4))

		    #ergonomics assessment here
			risk, self.main_angles = ergonomic_evaluation.scoring(keypoints_front,keypoints_side)

			if risk != 0:
		    #incremental average and variance
			
				if self.frame_No > 1:
					self._average = (self.old_average * (self.frame_No - 1) + self.main_angles) / self.frame_No 	#update mean
					self.S = self.S + (self.main_angles - self.old_average) * (self.main_angles - self._average)
					self.variance = self.S / self.frame_No  #update variance
			
				else:
					self.old_average = self.main_angles   #first frame
					self._average = self.main_angles
					
				
				#erogonomic assessment output
				#print("frame " +str(frame_id)"\n risk level: \n" + str(risk) + "\n", "average: \n" + str(self._average) + "\n", "variance: \n" + str(self.variance) + "\n")


				#print keypoints
				#print("Body keypoints: \n" + str(keypoints))
				#print("Face keypoints: \n" + str(self.datum.faceKeypoints))
				#print("Left hand keypoints: \n" + str(self.datum.handKeypoints[0]))
				#print("Right hand keypoints: \n" + str(self.datum.handKeypoints[1]))
				
				#cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", self.datum.cvOutputData)
				#cv2.waitKey(0)

				#save results images
				#cv2.imwrite('/home/student/result_frontside/'+str(self.frame_id)+'.jpg',self.datum.cvOutputData)
				
				
				with open('/home/student/result_frontside/result_frontknee.txt', 'a') as file_handle:
					file_handle.write('\nframe_id:')
					file_handle.write(str(self.frame_id))
					file_handle.write('\nrisk level:')
					file_handle.write(str(risk))
					file_handle.write('\naverage\n')
					file_handle.write(str(self._average))
					file_handle.write('\nvariance\n')
					file_handle.write(str(self.variance))


				#publish results
				self.publisher_pose.publish(self.br.cv2_to_imgmsg(np.array(self.datum.cvOutputData), "bgr8"))
				self.publisher_keypoints.publish(keypoints_whole)
				self.publisher_risk.publish(risk)
			

def main(args=None):
	rclpy.init(args=args)
	
	Pose_ergonomic = pose_ergonomic()
	rclpy.spin(Pose_ergonomic)  #loop

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	Pose_ergonomic.destroy_node()
	rclpy.shutdown()
 
if __name__ == '__main__':
	main()

