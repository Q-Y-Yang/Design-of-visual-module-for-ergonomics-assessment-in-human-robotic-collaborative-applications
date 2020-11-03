#!/usr/bin/env python
#!coding=utf-8

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

	#lookup table ABC
def lookup(UL,WW,N,TLE,MF1,MF2):
	#tableA = np.empty([19,9], dtype = int)
	#tableB = np.empty([7,13], dtype = int)
	#tableC = np.empty([9,8], dtype = int)

	tableA = np.array(([0,110,120,210,220,310,320,410,420],[11,1,2,2,2,2,3,3,3],[12,2,2,2,2,3,3,3,3],[13,2,3,3,3,3,3,4,4],[21,2,3,3,3,3,4,4,4],[22,3,3,3,3,3,4,4,4],[23,3,4,4,4,4,4,5,5],[31,3,3,4,4,4,4,5,5],[32,3,4,4,4,4,4,5,5],[33,4,4,4,4,4,5,5,5],[41,4,4,4,4,4,5,5,5],[42,4,4,4,4,4,5,5,5],[43,4,4,4,5,5,5,6,6],[51,5,5,5,5,5,6,6,7],[52,5,6,6,6,6,7,7,7],[53,6,6,6,7,7,7,7,8],[61,7,7,7,7,7,8,8,9],[62,8,8,8,8,8,9,9,9],[63,9,9,9,9,9,9,9,9]))

	tableB = np.array(([0,11,12,21,22,31,32,41,42,51,52,61,62],[10,1,3,2,3,3,4,5,5,6,6,7,7],[20,2,3,2,3,4,5,5,5,6,7,7,7],[30,3,3,3,4,4,5,5,6,6,7,7,7],[40,5,5,5,6,6,7,7,7,7,7,8,8],[50,7,7,7,7,7,8,8,8,8,8,8,8],[60,8,8,8,8,8,8,8,9,9,9,9,9]))

	tableC = np.array(([0,10,20,30,40,50,60,70],[100,1,2,3,3,4,5,5],[200,2,2,3,4,4,5,5],[300,3,3,3,4,4,5,6],[400,3,3,3,4,5,6,6],[500,4,4,4,5,6,7,7],[600,4,4,5,6,6,7,7],[700,5,5,6,6,7,7,7],[800,5,5,6,7,7,7,7]))
	scoreA = tableA[np.argwhere(tableA == UL)[0,0], np.argwhere(tableA == WW)[0,1]]
	scoreB = tableB[np.argwhere(tableB == N)[0,0], np.argwhere(tableB == TLE)[0,1]]
	print(scoreA, scoreB)
	WA = 0
	NLT = 0
	WA = (MF1 + scoreA)*100
	NTL = (MF2 + scoreB)*10
	if WA>800:
		WA = 800
	if NTL>70:
		NTL = 70

	scoreC = tableC[np.argwhere(tableC == WA)[0,0], np.argwhere(tableC == NTL)[0,1]]
	
	return scoreC

def main():
	#coordinates
	data = [4.54335022e+02,   1.82294769e+02 , 8.04848909e-01, 5.06112152e+02 ,  2.10155533e+02 ,  6.09847128e-01,  4.95443604e+02  , 2.07475998e+02 ,  5.00070214e-01, 5.06079437e+02 ,  3.00376770e+02 , 1.59751445e-01, 4.80851044e+02 ,  3.25624084e+02  , 6.74008787e-01, 5.16710571e+02 ,  2.15447479e+02  , 7.25321651e-01,  5.04807007e+02 ,  3.03014282e+02 , 8.44079137e-01,  4.27800171e+02 ,  2.95111786e+02 ,  8.12976718e-01,  5.35272278e+02,   3.76018341e+02 ,  4.74166840e-01,  5.45897461e+02,  3.76001495e+02,   3.74954462e-01,  5.51185486e+02 , 4.84854431e+02,   3.17633390e-01,  0.0 ,  0.0 ,  0.0,  5.25960693e+02 ,  3.76055206e+02 ,  4.73727733e-01,  4.74253937e+02 ,  4.84872131e+02 ,  3.76045287e-01,  0.0,   0.0,   0.0,  0.0 ,  0.0,  0.0,  4.60996216e+02 ,  1.67710571e+02 ,  8.66292834e-01,  0.0 , 0.0 , 0.0,  4.92849335e+02 ,  1.61047699e+02 ,  8.90019774e-01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ,0.0,0.0,0.0,0.0,0.0]
	hand = [151.80465698,295.50640869,0.64073163,161.76905823 , 285.10876465,    0.64167237,176.93228149 , 277.74377441 ,   0.74247301,194.69493103 , 275.5776062 ,    0.82608491,208.12521362,  274.71112061 ,   0.8844223,190.36257935,  288.14141846,    0.85013592,208.12521362,  285.54202271 ,   0.86302739,220.68902588 , 285.10876465 ,   0.91600031,230.65342712 , 285.54202271 ,   0.85338199,189.49610901  ,297.6725769   ,  0.80328763,208.12521362 , 295.93963623   , 0.90003651,221.55549622  ,295.50640869,    0.85108292,231.95314026,  295.93963623 ,   0.84625143,186.896698   , 305.4708252   ,  0.84855127, 204.65933228 , 305.4708252    , 0.85154438,217.22314453 , 305.4708252,     0.93645048,227.18754578 , 305.90405273,    0.81918281,185.59698486 , 311.96936035 ,   0.77913034,199.89373779 , 312.40258789  ,  0.88423419,207.69197083 ,312.40258789, 0.97999418,214.62373352,312.40,0.0]
	# read keypoints
	keypointsarr = np.array(data).reshape(25,3)	 #2D array for a person
	keypoints = np.delete(keypointsarr, -1, axis=1)   #delete confidence score
	handsarr = np.array(hand).reshape(21,3)	 #2D array for a person
	hand = np.delete(handsarr, -1, axis=1)
	#if keypoints[8,:]==[0,0] or keypoints[1,:]==[0,0]:
	#ros error

#RULA Score
	score1 = 0
	score2 = 0
	U = 0
	L = 0
	W1 = 0
	W2 = 0
	N = 0
	T = 0
	LE = 0
	UL = 0
	WW = 0
	N = 0
	TLE = 0
	MF1 = 0
	MF2 = 0

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
	angle3_1 = line_angle(keypoints[3,:], keypoints[4,:],keypoints[4,:], hand[9,:])

	#step9a neck twist or side bent
	angle9_1 = line_angle(keypoints[0,:], keypoints[1,:], keypoints[8,:], keypoints[1,:])

	#step10 trunk side bend
	angle10_1 = abs(line_angle(keypoints[8,:],keypoints[1,:], keypoints[1,:], keypoints[1,:]+[1, 0]) - 90)

	#step10a trunk twist
	angle10_2 = line_angle(keypoints[2,:], keypoints[8,:], keypoints[5,:], keypoints[8,:])

	#step11 legs evenly-balanced
	angle11_1 = line_angle(keypoints[9,:], keypoints[10,:], keypoints[10,:], keypoints[11,:])
	angle11_2 = line_angle(keypoints[12,:], keypoints[13,:], keypoints[13,:], keypoints[14,:])


	#scoring
	#step1
	if abs(angle11)<20:
		U = U+1
	elif angle11<-20 or angle11>20 and angle11<45:
		U = U+2
	elif angle11>45 and angle11<90:
		U = U+3
	elif angle11>90:
		U = U+4

	#step2 & 2a
	if angle21>80 and angle21<100:
		L = L+1
	elif angle21>100 or angle21<80 and angle21>5:
		L = L+2

	if angle2_1>5 or angle2_1<-5:
		L =  L+1
	if angle2_2>5 or angle2_2<-5:
		L =  L+1

	#step3 & 3a
	if angle31<2 or angle31>-2:
		W1 = W1+1
	elif angle31<12 or angle31>-12:
		W1 = W1+2
	elif angle31>15 or angle31<-15:
		W1 = W1+3

	if angle3_1>5 or angle3_1<-5:
		W1 = W1+1

	#step4
	if angle41>30:
		W2 =  W2+1
	else:
		W2 = W2+2

	#step9 & 9a
	if angle91>0 and angle91<10:
		N = N+1
	elif angle91>10 and angle91<20:
		N = N+2
	elif angle91>20:
		N =N+3
	elif angle91<0:
		N = N+4

	if angle9_1>5 or angle9_1<-5:
		N = N+1

	#step10 & 10a
	if angle101==0:
		T = T+1
	elif angle101>0 and angle101<20:
		T = T+2
	elif angle101>20 and angle101<60:
		T = T+3
	elif angle101>60:
		T = T+4

	if angle10_2<30:
		T =  T+1

	#step11 
	if angle11_1>170 and angle11_2>170:
		LE = LE+1
	else: LE = LE+2
	#tableA = np.zeros([19,9], dtype = int)
	#tableB = np.empty([7,13], dtype = int)
	#tableC = np.empty([9,8], dtype = int)
	#lookup table A
	UL = U*10 + L
	WW = W1*100 + W2*10
	N = N*10
	TLE = T*10 + LE
	MF1 = 1
	MF2 = 1

	risklevel = lookup(UL,WW,N,TLE,MF1,MF2)
	print(risklevel,U,L,W1,W2,N/10,T,LE)

if __name__ == '__main__':
    main()
