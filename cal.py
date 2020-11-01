#!/usr/bin/env python
#!coding=utf-8

import cv2
import numpy as np
import math

# calculate angle between two lines
def line_angle(A1, A2, B1, B2):
	kline1 = (A2[1] - A1[1])/(A2[0] - A1[0])
	kline2 = (B2[1] - B1[1])/(B2[0] - B1[0])
	tan_k = (kline2 - kline1)/(1 + kline2 * kline1)
	arctan = math.atan(tan_k)
	angle = arctan * 180 / 3.1415926
	return angle

def main():
	#coordinates
	A1 = np.array([0,0])
	A2 = np.array([2,0])
	B1 = np.array([0,0])
	B2 =np.array([1,1.732])
	data = [281.1230773925781, 172.7841033935547, 0.5517820715904236, 235.4836883544922, 222.3551025390625, 0.680304229259491, 248.4776153564453, 227.59487915039062, 0.6968498229980469, 262.8547668457031, 311.1439514160156, 0.7227026224136353, 278.544189453125, 384.2109375, 0.7792686223983765, 223.68553161621094, 213.26657104492188, 0.485501766204834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 231.50283813476562, 360.7401123046875, 0.5020049214363098, 243.28176879882812, 363.3276672363281, 0.4808925986289978, 244.55772399902344, 462.537353515625, 0.29941526055336, 0.0, 0.0, 0.0, 217.16526794433594, 359.4183654785156, 0.43657034635543823, 221.05130004882812, 456.01434326171875, 0.29060328006744385, 0.0, 0.0, 0.0, 275.95330810546875, 170.16468811035156, 0.7637776136398315, 0.0, 0.0, 0.0, 256.3580627441406, 183.26910400390625, 0.9193944931030273, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	# read keypoints
	keypointsarr = np.array(data).reshape(25,3)	 #2D array for a person
	keypoints = np.delete(keypointsarr, -1, axis=1)   #delete confidence score
	#if keypoints[8,:]==[0,0] or keypoints[1,:]==[0,0]:
	#ros error

#RULA Score
	score1 = 0
	score2 = 0

	#side view
	#step1 upper arm
	angle11 = line_angle(keypoints[2,:], keypoints[3,:], keypoints[8,:], keypoints[1,:])  #normally from side view both shoulders can be detected.
	#angle12 = line_angle(keypoints[5], keypoints[6], keypoints[8], keypoints[1])

	#step1a shoulder raised?
	angle13 = line_angle(keypoints[2,:], keypoints[1,:],keypoints[8,:], keypoints[1,:])  

	#step2 lower arm
	angle21 = line_angle(keypoints[3,:], keypoints[4,:], keypoints[2,:], keypoints[3,:])

	#step3 wrist  also need hand points
	angle31 = line_angle(keypoints[3,:], keypoints[4,:], keypoints[4,:], hand[9,:])
	#step4 wrist twist
	angle41 = line_angle(hand[0,:], hand[9,:], hand[0,:], hand[2,:])
	#step9 neck
	angle91 = line_angle(keypoints[8,:], keypoints[1,:], keypoints[1,:], keypoints[17,:])

	#step10 trunk
	angle101 = abs(line_angle(keypoints[8,:],keypoints[1,:], keypoints[1,:], keypoints[1,:]+[1, 0]) - 90)

	#from top front view
	#step1a abduction of shoulder
	angle1_1 = line_angle(keypoints[2,:], keypoints[3,:], keypoints[8,:], keypoints[1,:])
	angle1_2 = line_angle(keypoints[5,:], keypoints[6,:], keypoints[8,:], keypoints[1,:])

	#step2a lower arm outside of body
	angle2_1 = line_angle(keypoints[3,:], keypoints[4,:], keypoints[8,:], keypoints[1,:])
	angle2_2 = line_angle(keypoints[6,:], keypoints[7,:], keypoints[8,:], keypoints[1,:])

	#step3a wrist bent from midline
	angle3_1 = line_angle((keypoints[3,:], keypoints[4,:],keypoints[4,:], hand[9,:])

	#step9a neck twist or side bent
	angle9_1 = line_angle(keypoints[0,:], keypoints[1,:], keypoints[8,:], keypoints[1,:])

	#step10 trunk side bend
	angle10_1 = abs(line_angle(keypoints[8,:],keypoints[1,:], keypoints[1,:], keypoints[1,:]+[1,0]) - 90)

	#step10a trunk twist
	angle10_2 = line_angle(keypoints[2,:], keypoints[8,:], keypoints[5,:], keypoints[8,:])

	#step11 legs evenly-balanced
	angle11_1 = line_angle(keypoints[9,:], keypoints[10,:], keypoints[10,:], keypoints[11,:])
	angle11_2 = line_angle(keypoints[12,:], keypoints[13,:], keypoints[13,:], keypoints[14,:])


	#scoring
	#step1
	if angle11<20 and angle11>-20:
		score1 +=score1
	elif angle11<-20 or angle11>20 and angle11<45:
		score1 = score1+2
	elif angle11>45 and angle11<90:
		score1 = score1+3
	elif angle11>90:
		score1 = score1+4

	#step2 & 2a
	if angle21>80 and angle21<100:
		score1 +=score1
	elif angle21>100 or angle21<80 and angle21>5:
		score1 = score1+2

	if angle2_1>5 or angle2_1<-5:
		score1 += score1
	if angle2_2>5 or angle2_2<-5:
		score1 += score1

	#step3 & 3a
	if angle31<2 or angle31>-2:
		score1 +=score1
	elif angle31<12 or angle31>-12:
		score1 = score1+2
	elif angle31>15 or angle31<-15:
		score1 = score1+3

	if angle3_1>5 or angle3_1<-5:
		score1 +=score1

	#step9 & 9a
	if angle91>0 and angle91<10:
		score2 +=score2
	elif angle91>10 and angle91<20:
		score2 = score2+2
	elif angle91>20:
		score2 =score2+3
	elif angle91<0:
		score2 = score2+4

	if angle9_1>5 or angle9_1<-5:
		score2 +=score2

	#step10 & 10a
	if angle101==0:
		score2 +=score2
	elif angle101>0 and angle101<20:
		score2 = score2+2
	elif angle101>20 and angle101<60:
		score2 = score2+3
	elif angle101>60:
		score2 = score2+4

	#if angle10_2

	#step11 
	if angle11_1>170 and angle11_2>170:
		score2 +=score2
	else score2 = score2+2

	print(angle11,angle13,angle21,angle91,angle101,angle1_1,angle1_2,angle9_1,angle10_1,angle11_1,angle11_2)

if __name__ == '__main__':
    main()
