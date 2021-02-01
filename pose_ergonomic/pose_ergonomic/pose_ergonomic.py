#!/usr/bin/env python3
#!coding=utf-8
 

#function: 
#Subscribe synchronized frames from different cameras
#Detect body and hands keypoints by OpenPose python API
#Performing ergonomics assessment
#inputs: two synchronized image messages with header including timestamp captured by two cameras from side view and front-top view
#outputs: Image with keypoints annotation, joint angles, risk level of ergonomic assessment 
 
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int64, Float32, Float64MultiArray
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
from . import ergonomic_evaluation, arm_ergonomic, ergonomic_nerpa
import time



class pose_ergonomic(Node):

    
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
			parser.add_argument("--result_path", default="/home/student/result/", help="set a path to save results")
			parser.add_argument("--single_view", default=False, type= bool, help="single view or two views")
			parser.add_argument("--rot90", default=False, type= bool, help="clockwise rotate 90 degrees")
			parser.add_argument("--camera_topic", default="/syn_img", help="choose a topic as input image")
			parser.add_argument("--save_result", default= True, type= bool, help="save result images")
			parser.add_argument("--ergonomic", default= 1, type= int, help="select 1 for RULA, 2 for NERPA")
			args = parser.parse_args()
			self.single_view = args.single_view
			self.result_path = args.result_path
			self.rot90 = args.rot90
			self.camera_topic = args.camera_topic
			self.save_result = args.save_result
			self.ergonomic = args.ergonomic
			# Custom Params (refer to include/openpose/flags.hpp for more parameters)
			params = dict()
			params["model_folder"] = "/home/student/openpose/models/"
			params["face"] = True
			params["hand"] = True


			#self.result_path = args.result_path
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
		print('Openpose started')
		#subscriber and publisher initialization
		#input subscriber
		self.br = CvBridge()
		self.subscription_synimg = self.create_subscription(Image, self.camera_topic, self.callback,10)
		
		#output publisher
		self.publisher_pose = self.create_publisher(Image,'/pose',10)	#images with keypoints annotation
		#self.publisher_keypoints = self.create_publisher(Float32MultiArray,'/keypoints',10)	#keypoints coordinates
		self.publisher_risk = self.create_publisher(Int64,'/risk',10)	#risk level
		self.publisher_angles = self.create_publisher(Float32MultiArray,'/angles',10)
		#self.subscription   #prevent unused variable warning
		self.current_timestamp = 1     #timestamp init
		self.main_angles = np.empty((1,4),dtype = float)    #to store some calculated angle results
		self.angles =  np.empty((1,20),dtype = float)
       #average calculation
		self.old_average = np.empty((1,4),dtype = float)     
		self._average = np.empty((1,4),dtype = float) 

		self.frame_No = 0      #frame number init in each one minute
		self.frame_id = 0	#count frames

       #variance calculation
		self.S = np.zeros((1,4),dtype = float) 
		self.variance = np.zeros((1,4),dtype = float) 

		self.load = np.zeros((1,2),dtype = float)
		self.old_load = np.zeros((1,2),dtype = float)
		self.aver_load = np.zeros((1,2),dtype = float)   

	def callback(self,data):
		#input image
		print(self.frame_No)
		start = time.time()
		try:
			if self.rot90 is True:
				cv_img = cv2.flip(cv2.transpose(self.br.imgmsg_to_cv2(data, "bgr8")), 0)
			else:
				cv_img = self.br.imgmsg_to_cv2(data, "bgr8")
			
			self.frame_No = self.frame_No + 1
			self.frame_id = self.frame_id + 1
			self.old_average = self._average
			self.old_load = self.load
			self.datum.cvInputData = cv_img	#imageToProcess
		except CvBridgeError as e:
			print(e)
		self.opWrapper.emplaceAndPop([self.datum])		
		
		# array init
		keypoints_whole = Float32MultiArray()
		risk = Int64()
		angles = Float32MultiArray()
		#hands = Float32MultiArray()
	
		#check if detect at least a person and a hand, if no, exist
		try:
			[i,j,k] = np.array(self.datum.poseKeypoints).shape  #i persons, j keypoints, k=3
		except ValueError:
			print('ERROR: No one detected!')
			i = 0	
			#sys.exit(-1)	#exist
		try:
			[l,m,n,r] = np.array(self.datum.handKeypoints).shape	#l hands, m persons, n keypoints, r=3
		except ValueError:
			print('ERROR: Hand not detected!')
			i = 0
			#sys.exit(-1)	#exist
		#if i !=2:
		#cv2.imwrite('/home/student/arm/'+str(self.frame_id)+'.jpg',self.datum.cvOutputData)
		#check if there is two persons
		#cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", self.datum.cvOutputData)
		#cv2.waitKey(0)	
		
		if self.single_view is False and i == 2:
	
			#save datum.poseKeypoints in a numpy array
			#body = np.zeros(i*j*k)
			#hand = np.zeros(l*m*n*r)
			body = np.array(self.datum.poseKeypoints)
			hands = np.array(self.datum.handKeypoints)
			
			keypoints_side = np.concatenate((body[0],hands[0][0],hands[0][1]))
			keypoints_front = np.concatenate((body[1],hands[1][0],hands[1][1]))
			#tuple! to publish array msgs
			whole_body = np.array(np.concatenate((keypoints_side, keypoints_front))).reshape(134,3)
			#tup_wb = tuple(whole_body)
			#print(whole_body)
			#keypoints_whole.data = tup_wb
			#self.publisher_keypoints.publish(keypoints_whole)

			if data.header.stamp.sec  > self.current_timestamp + 60:  #check one minute
				self.current_timestamp = data.header.stamp.sec
		 
		        #reset after one minute
				self.frame_No = 0

				if np.sum(self.variance) < 130:
					print('Alert: too static in one minute')
				self.variance = self.old_average = self._average = self.S = np.zeros((1,4))

				if self.aver_load.any() > 4.4 and self.aver_load.any() < 22:
					print('Alert: repeated load')
				if self.aver_load.any() > 22:
					print('Alert: repeated heavy load')

		    #ergonomics assessment here
			if keypoints_front is not None and keypoints_side is not None:
				if self.ergonomic == 1:
					ergonomic = ergonomic_evaluation.scoring(keypoints_front,keypoints_side,self.load)
				elif self.ergonomic == 2:
					ergonomic = ergonomic_nerpa.scoring(keypoints_front,keypoints_side,self.load)
				
				self.angles = ergonomic[2]
				self.main_angles = ergonomic[1]
				self.risklevel = ergonomic[0]
			
			else:
				print('Not sufficient input keypoints')

			end = time.time()
			fps = 1 / (end - start)
			print('FPS:'+str(fps))
			
			if self.risklevel != 0:
		    #incremental average and variance
			
				if self.frame_No > 1:
					self._average = (self.old_average * (self.frame_No - 1) + self.main_angles) / self.frame_No 	#update mean
					self.S = self.S + (self.main_angles - self.old_average) * (self.main_angles - self._average)
					self.variance = self.S / self.frame_No  #update variance
					self.aver_load = (self.old_load * (self.frame_No - 1) + self.load) / self.frame_No
				else:
					self.old_average = self.main_angles   #first frame
					self._average = self.main_angles
					self.old_load = self.load
					self.aver_load = self.load
						#for muscle strain in one minute evaluation

				#erogonomic assessment output
				#print("frame " +str(frame_id)"\n risk level: \n" + str(risk) + "\n", "average: \n" + str(self._average) + "\n", "variance: \n" + str(self.variance) + "\n")


				#print keypoints
				#print("Body keypoints: \n" + str(keypoints))
				#print("Face keypoints: \n" + str(self.datum.faceKeypoints))
				#print("Left hand keypoints: \n" + str(self.datum.handKeypoints[0]))
				#print("Right hand keypoints: \n" + str(self.datum.handKeypoints[1]))
				risk.data = self.risklevel.item()
				angles.data = tuple(self.angles)
				self.publisher_risk.publish(risk)
				self.publisher_angles.publish(angles)

				#save results images
				if self.save_result is True:
					with open(str(self.result_path) + 'results.txt', 'a') as file_handle:
						file_handle.write('\nframe_id:')
						file_handle.write(str(self.frame_id))
						file_handle.write('\nrisk level:')
						file_handle.write(str(self.risklevel))
						file_handle.write('\naverage\n')
						file_handle.write(str(self._average))
						file_handle.write('\nvariance\n')
						file_handle.write(str(self.variance))
					position = (50,50)
					txt = 'risk:'+str(risk)+'\nFPS:'+str(fps)
					cv2.putText(self.datum.cvOutputData,txt, position, cv2.FONT_HERSHEY_SIMPLEX, 6, (255,255, 255), 25)

				



		elif self.single_view is True and i == 1:
			
			body = np.array(self.datum.poseKeypoints)
			hands = np.array(self.datum.handKeypoints)
			keypoints_side = np.concatenate((body[0],hands[0][0]))  #body and only one hand here
		
		
		    #ergonomics assessment here
			if keypoints_side is not None:
				upperarm, lowerarm = arm_ergonomic.scoring(keypoints_side)
				if self.save_result is True:
					with open(str(self.result_path) + 'results.txt', 'a') as file_handle:
						file_handle.write('\nframe_id:')
						file_handle.write(str(self.frame_id))
						file_handle.write('\nupperarm:')
						file_handle.write(str(upperarm))
						file_handle.write('\nlowerarm\n')
						file_handle.write(str(lowerarm))
			else:
				print('Not sufficient input keypoints')

			end = time.time()
			fps = 1 / (end - start)
			
			print('FPS:'+str(fps))

		if self.save_result is True:

			cv2.imwrite(str(self.result_path) +str(self.frame_id)+'.jpg',self.datum.cvOutputData)

				#publish results
		self.publisher_pose.publish(self.br.cv2_to_imgmsg(np.array(self.datum.cvOutputData), "bgr8"))
				
				
			

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

