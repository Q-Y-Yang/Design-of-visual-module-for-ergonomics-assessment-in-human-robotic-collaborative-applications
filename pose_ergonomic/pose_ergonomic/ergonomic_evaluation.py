import numpy as np
import math

def line_angle(A1, A2, B1, B2):

	if (A2[0] - A1[0]) != 0 and (B2[0] - B1[0]) != 0:
		kline1 = (A2[1] - A1[1])/(A2[0] - A1[0])
		kline2 = (B2[1] - B1[1])/(B2[0] - B1[0])
		tan_k = (kline2 - kline1)/(1 + kline2 * kline1)
		arctan = math.atan(tan_k)
		angle = arctan * 180 / 3.1415926
		return angle
	else:
		return 500




def lookup(UL,WW,N,TLE,MF1,MF2):
	tableA = np.array(([0,110,120,210,220,310,320,410,420],[11,1,2,2,2,2,3,3,3],[12,2,2,2,2,3,3,3,3],[13,2,3,3,3,3,3,4,4],[21,2,3,3,3,3,4,4,4],[22,3,3,3,3,3,4,4,4],[23,3,4,4,4,4,4,5,5],[31,3,3,4,4,4,4,5,5],[32,3,4,4,4,4,4,5,5],[33,4,4,4,4,4,5,5,5],[41,4,4,4,4,4,5,5,5],[42,4,4,4,4,4,5,5,5],[43,4,4,4,5,5,5,6,6],[51,5,5,5,5,5,6,6,7],[52,5,6,6,6,6,7,7,7],[53,6,6,6,7,7,7,7,8],[61,7,7,7,7,7,8,8,9],[62,8,8,8,8,8,9,9,9],[63,9,9,9,9,9,9,9,9]))

	tableB = np.array(([0,11,12,21,22,31,32,41,42,51,52,61,62],[10,1,3,2,3,3,4,5,5,6,6,7,7],[20,2,3,2,3,4,5,5,5,6,7,7,7],[30,3,3,3,4,4,5,5,6,6,7,7,7],[40,5,5,5,6,6,7,7,7,7,7,8,8],[50,7,7,7,7,7,8,8,8,8,8,8,8],[60,8,8,8,8,8,8,8,9,9,9,9,9]))

	tableC = np.array(([0,10,20,30,40,50,60,70],[100,1,2,3,3,4,5,5],[200,2,2,3,4,4,5,5],[300,3,3,3,4,4,5,6],[400,3,3,3,4,5,6,6],[500,4,4,4,5,6,7,7],[600,4,4,5,6,6,7,7],[700,5,5,6,6,7,7,7],[800,5,5,6,7,7,7,7]))
	scoreA = tableA[np.argwhere(tableA == UL)[0,0], np.argwhere(tableA == WW)[0,1]]
	scoreB = tableB[np.argwhere(tableB == N)[0,0], np.argwhere(tableB == TLE)[0,1]]
	#print(scoreA, scoreB)
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

def scoring(keypoints_front, keypoints_side):
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
	#keypoints = np.array(msg.data).reshape(201,3)	 #2D array for a person
	#keypoints = np.delete(keypoints, -1, axis=1)   #delete confidence score
	keypoints_side = np.delete(keypoints_side, -1, axis=1)
	keypoints_front = np.delete(keypoints_front, -1, axis=1)
	#side view
	#step1 upper arm
	angle_uarm = line_angle(keypoints_side[2,:], keypoints_side[3,:], keypoints_side[8,:], keypoints_side[1,:])  #normally from side view both shoulders can be detected.
		#angle12 = line_angle(keypoints[5], keypoints[6], keypoints[8], keypoints[1])

	#step1a shoulder raised?
	angle_shoulder = line_angle(keypoints_side[2,:], keypoints_side[1,:],keypoints_side[8,:], keypoints_side[1,:])  

	#step2 lower arm
	angle_larm = line_angle(keypoints_side[3,:], keypoints_side[4,:], keypoints_side[2,:], keypoints_side[3,:])

	#step3 wrist  also need hand points
	angle_wrist = line_angle(keypoints_side[3,:], keypoints_side[4,:], keypoints_side[4,:], keypoints_side[34,:])
	#step4 wrist twist
	angle_wristtw = line_angle(keypoints_side[25,:], keypoints_side[34,:], keypoints_side[25,:], keypoints_side[27,:])
	#step9 neck
	angle_neck = line_angle(keypoints_side[8,:], keypoints_side[1,:], keypoints_side[1,:], keypoints_side[17,:])

	#step10 trunk
	angle_trunk = abs(line_angle(keypoints_side[8,:],keypoints_side[1,:], keypoints_side[1,:], keypoints_side[1,:]+[1, 0]) - 90)

	#from top front view
	#step1a abduction of shoulder
	angle_shou_abduct = line_angle(keypoints_front[2,:], keypoints_front[3,:], keypoints_front[8,:], keypoints_front[1,:])
	angle_shou_abduct2 = line_angle(keypoints_front[5,:], keypoints_front[6,:], keypoints_front[8,:], keypoints_front[1,:])

	#step2a lower arm outside of body
	angle_larm_out = line_angle(keypoints_front[3,:], keypoints_front[4,:], keypoints_front[8,:], keypoints_front[1,:])
	angle_larm_out2 = line_angle(keypoints_front[6,:], keypoints_front[7,:], keypoints_front[8,:], keypoints_front[1,:])

	#step3a wrist bent from midline
	angle_wristb = line_angle(keypoints_front[3,:], keypoints_front[4,:],keypoints_front[4,:], keypoints_front[34,:])

	#step9a neck twist or side bent
	angle_necktw = line_angle(keypoints_front[0,:], keypoints_front[1,:], keypoints_front[8,:], keypoints_front[1,:])

	#step10 trunk side bend
	angle_trunkb = abs(line_angle(keypoints_front[8,:],keypoints_front[1,:], keypoints_front[1,:], keypoints_front[1,:]+[1, 0]) - 90)

	#step10a trunk twist
	angle_trunktw = line_angle(keypoints_front[2,:], keypoints_front[8,:], keypoints_front[5,:], keypoints_front[8,:])

	#step11 legs evenly-balanced
	angle_leg1 = line_angle(keypoints_front[9,:], keypoints_front[10,:], keypoints_front[10,:], keypoints_front[11,:])
	angle_leg2 = line_angle(keypoints_front[12,:], keypoints_front[13,:], keypoints_front[13,:], keypoints_front[14,:])


	#scoring
	#step1 &1a
	if abs(angle_uarm)<20:
		U = U+1
	elif angle_uarm<-20 or angle_uarm>20 and angle_uarm<45:
		U = U+2
	elif angle_uarm>45 and angle_uarm<90:
		U = U+3
	elif angle_uarm>90 and angle_uarm<500:
		U = U+4
	

	if angle_shou_abduct > 45 or angle_shou_abduct2 >45:
		U = U+1

	if angle_shoulder > 90:
		U = U+1

	#step2 & 2a
	if angle_larm>80 and angle_larm<100:
		L = L+1
	elif angle_larm>100 or angle_larm<80 and angle_larm>5:
		L = L+2

	if angle_larm_out>5 or angle_larm_out<-5:
		L =  L+1
	if angle_larm_out2>5 or angle_larm_out2<-5:
		L =  L+1
	if L > 3:
		L = 3
	#step3 & 3a
	if angle_wrist<2 or angle_wrist>-2:
		W1 = W1+1
	elif angle_wrist<12 or angle_wrist>-12:
		W1 = W1+2
	elif angle_wrist>15 or angle_wrist<-15:
		W1 = W1+3

	if angle_wristb>5 or angle_wristb<-5:
		W1 = W1+1

	#step4
	if angle_wristtw>30:
		W2 =  W2+1
	else:
		W2 = W2+2

	#step9 & 9a
	if angle_neck>0 and angle_neck<10:
		N = N+1
	elif angle_neck>10 and angle_neck<20:
		N = N+2
	elif angle_neck>20:
		N =N+3
	elif angle_neck<0:
		N = N+4

	if angle_necktw>5 or angle_necktw<-5:
		N = N+1

	#step10 & 10a
	if angle_trunk==0:
		T = T+1
	elif angle_trunk>0 and angle_trunk<20:
		T = T+2
	elif angle_trunk>20 and angle_trunk<60:
		T = T+3
	elif angle_trunk>60:
		T = T+4

	if angle_trunktw<30:
		T =  T+1

	if angle_trunkb < 40:
		T = T+1

	#step11 
	if angle_leg1>170 and angle_leg2>170:
		LE = LE+1
	else: LE = LE+2
	if U == 0 or L== 0 or W1== 0 or W2== 0 or N== 0 or T== 0 or LE == 0:
		return 0, [0, 0, 0]	
	#lookup table A
	UL = U*10 + L
	WW = W1*100 + W2*10
	N = N*10
	TLE = T*10 + LE
	MF1 = 1
	MF2 = 1

	risklevel = lookup(UL,WW,N,TLE,MF1,MF2)
	
	main_angles = np.asarray([angle_uarm, angle_larm, angle_neck, angle_trunk])

	return risklevel, main_angles

