import numpy as np
import math

def line_angle(A1, A2, B1, B2):
	cosine = np.dot(A1 - A2, B1 - B2) / (
                    np.linalg.norm(A1 - A2, ord=2) * np.linalg.norm(B1 - B2, ord=2))
	if cosine > -1 and cosine < 1:
		arccos = math.acos(cosine)
		angle = arccos * 180 / 3.1415926
	else:
		angle = 0
	return angle




def lookup(UL,WW,N,TLE,F1,F2):
	tableA = np.array(([0,110,120,210,220,310,320,410,420],[11,1,2,2,2,2,3,3,3],[12,2,2,2,2,3,3,3,3],[13,2,3,3,3,3,3,4,4],[21,2,3,2,3,3,3,4,4],[22,3,3,3,3,3,3,4,4],[23,3,3,4,4,4,4,5,5],[31,3,3,4,4,4,4,5,5],[32,3,4,4,4,4,4,5,5],[33,4,4,4,4,4,5,5,5],[41,4,4,4,4,4,5,5,5],[42,4,4,4,4,4,5,5,5],[43,4,4,4,5,5,5,6,6],[51,5,5,5,5,5,6,6,7],[52,5,6,6,6,6,7,7,7],[53,6,6,6,7,7,7,7,8]))

	tableB = np.array(([0,11,12,21,22,31,32,41,42,51,52,61,62],[10,1,3,2,3,3,4,5,5,6,6,7,7],[20,2,3,2,3,4,5,5,5,6,7,7,7],[30,3,3,3,4,4,5,5,6,6,7,7,7],[40,5,5,5,6,6,7,7,7,7,7,8,8],[50,7,7,7,7,7,8,8,8,8,8,8,8],[60,8,8,8,8,8,8,9.,9,9,9,9,9]))

	tableC = np.array(([0,10,20,30,40,50,60,70],[100,1,2,3,3,4,5,5],[200,2,2,3,4,4,5,5],[300,3,3,3,4,4,5,6],[400,3,3,3,4,5,6,6],[500,4,4,4,5,6,7,7],[600,4,4,5,6,6,7,7],[700,6,6,6,6,7,7,7],[800,6,6,6,7,7,7,7]))
	scoreA = tableA[np.argwhere(tableA == UL)[0,0], np.argwhere(tableA == WW)[0,1]]
	scoreB = tableB[np.argwhere(tableB == N)[0,0], np.argwhere(tableB == TLE)[0,1]]
	#print(scoreA, scoreB)
	WA = 0
	NLT = 0
	WA = (F1 + scoreA)*100
	NTL = (F2 + scoreB)*10
	if WA>800:
		WA = 800
	if NTL>70:
		NTL = 70

	scoreC = tableC[np.argwhere(tableC == WA)[0,0], np.argwhere(tableC == NTL)[0,1]]
	
	return scoreC

def scoring(keypoints_front, keypoints_side, load):

	#intermediate scores init
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
	F1 = 0
	F2 = 0
	#keypoints = np.array(msg.data).reshape(201,3)	 #2D array for a person
	#keypoints = np.delete(keypoints, -1, axis=1)   #delete confidence score
	keypoints_side = np.delete(keypoints_side, -1, axis=1)
	keypoints_front = np.delete(keypoints_front, -1, axis=1)

	keypoints_side_zeros = np.array(np.where(keypoints_side[:,0]==0.0))
	keypoints_front_zeros = np.array(np.where(keypoints_front[:,0]==0.0))
	#check main keypoints coordinates
	if 1 in keypoints_side_zeros or 8 in keypoints_side_zeros or 1 in keypoints_front_zeros or 8 in keypoints_front_zeros or 4 in keypoints_front_zeros or 7 in keypoints_front_zeros:
		return 0, [0, 0, 0], np.zeros((1,20), dtype = float)		#if one of main keypoints is missing, stop ergonomic assessment
 
	#side view
	#step1 upper arm
	if 3 in keypoints_side_zeros or 2 in keypoints_side_zeros:
		angle_uarmr = 0
	else:
		angle_uarmr = line_angle(keypoints_side[2,:], keypoints_side[3,:], keypoints_side[1,:], keypoints_side[8,:])  #right upper arm
	if 5 in keypoints_side_zeros or 6 in keypoints_side_zeros:
		angle_uarml = 0
	else:
		angle_uarml = line_angle(keypoints_side[5,:], keypoints_side[6,:], keypoints_side[1,:], keypoints_side[8,:])		#left upper arm
	if max(angle_uarmr,angle_uarml) > 60:	#score3
		angle_uarm = max(angle_uarmr,angle_uarml)
	else: 	#score 1-2
		angle_uarm = min(angle_uarmr,angle_uarml)

	#step1a shoulder raised?
	if 2 in keypoints_side_zeros:
		angle_shoulderr = 0
	else:
		angle_shoulderr = line_angle(keypoints_side[2,:], keypoints_side[1,:],keypoints_side[8,:], keypoints_side[1,:])  #right shoulder
	if 5 in keypoints_side_zeros:
		angle_shoulderl = 0
	else:	
		angle_shoulderl = line_angle(keypoints_side[5,:], keypoints_side[1,:],keypoints_side[8,:], keypoints_side[1,:])  #left shoulder
	angle_shoulder = max(angle_shoulderr,angle_shoulderl)

	#step2 lower arm
	if angle_uarmr ==0 or 4 in keypoints_side_zeros:
		angle_larmr = 0
	else:
		angle_larmr = line_angle(keypoints_side[3,:], keypoints_side[4,:], keypoints_side[1,:], keypoints_side[8,:])
	if angle_uarml ==0 or 7 in keypoints_side_zeros:
		angle_larml = 0
	else:
		angle_larml = line_angle(keypoints_side[6,:], keypoints_side[7,:], keypoints_side[1,:], keypoints_side[8,:])
	angle_larm = max(abs(angle_larmr),abs(angle_larml))

	#step3 wrist  also need hand points
	if 34 in keypoints_side_zeros or angle_larml ==0:
		angle_wristl = 0
	else: 
		angle_wristl = line_angle(keypoints_side[6,:], keypoints_side[7,:], keypoints_side[4,:], keypoints_side[34,:])	#left wrist
	if 43 in keypoints_side_zeros or angle_larmr ==0:
		angle_wristr = 0
	else: 
		angle_wristr = line_angle(keypoints_side[3,:], keypoints_side[4,:], keypoints_side[4,:], keypoints_side[43,:])
	angle_wrist = max(abs(angle_wristl), abs(angle_wristr))
	#step9 neck
	if 17 in keypoints_side_zeros:
		angle_neck = 0
	else:
		angle_neck = line_angle(keypoints_side[8,:], keypoints_side[1,:], keypoints_side[1,:], keypoints_side[17,:])

	#step10 trunk
	angle_trunk = abs(180 - line_angle(keypoints_side[8,:],keypoints_side[1,:], keypoints_side[1,:], keypoints_side[1,:]+[0, 1]))

	#step11 legs evenly-balanced
	if 9 in keypoints_side_zeros or 10 in keypoints_side_zeros or 11 in keypoints_side_zeros:
		angle_legr = 0
	else:
		angle_legr = line_angle(keypoints_side[9,:], keypoints_side[10,:], keypoints_side[10,:], keypoints_side[11,:])
	if 12 in keypoints_side_zeros or 13 in keypoints_side_zeros or 14 in keypoints_side_zeros:
		angle_legl = 0
	else:
		angle_legl = line_angle(keypoints_side[12,:], keypoints_side[13,:], keypoints_side[13,:], keypoints_side[14,:])
		

	#from top front view
	#step1a abduction of shoulder
	if 3 in keypoints_front_zeros or 2 in keypoints_front_zeros:
		angle_shou_abductr = 0
	else:
		angle_shou_abductr = line_angle(keypoints_front[2,:], keypoints_front[3,:], keypoints_front[1,:], keypoints_front[8,:])
	if 5 in keypoints_front_zeros or 6 in keypoints_front_zeros:
		angle_shou_abductl = 0
	else:
		angle_shou_abductl = line_angle(keypoints_front[5,:], keypoints_front[6,:], keypoints_front[1,:], keypoints_front[8,:])
	angle_shou_abduct = max(angle_shou_abductr, angle_shou_abductl)

	#step2a lower arm outside of body
	if 3 in keypoints_front_zeros or 4 in keypoints_front_zeros:
		angle_larm_outr = 0
	else:
		angle_larm_outr = line_angle(keypoints_front[3,:], keypoints_front[4,:], keypoints_front[1,:], keypoints_front[8,:])
	if 6 in keypoints_front_zeros or 7 in keypoints_front_zeros:
		angle_larm_outl = 0
	else:
		angle_larm_outl = line_angle(keypoints_front[6,:], keypoints_front[7,:], keypoints_front[1,:], keypoints_front[8,:])
	angle_larm_out = max(angle_larm_outr, angle_larm_outl)

	#step3a wrist bent from midline 
	if 34 in keypoints_front_zeros or angle_larm_outl ==0:
		angle_wristbl = 0
	else:
		angle_wristbl = line_angle(keypoints_front[6,:], keypoints_front[7,:],keypoints_front[7,:], keypoints_front[34,:])	#left wrist
	if 43 in keypoints_front_zeros or angle_larm_outr ==0:
		angle_wristbr = 0
	else:
		angle_wristbr = line_angle(keypoints_front[3,:], keypoints_front[4,:],keypoints_front[4,:], keypoints_front[55,:])
	angle_wristb = max(angle_wristbr, angle_wristbl)


	#step4 wrist twist
	if 25 in keypoints_front_zeros or 42 in keypoints_front_zeros or 26 in keypoints_front_zeros:
		angle_wristtwl = 0
	else:
		angle_wristtwl = line_angle(keypoints_front[25,:], keypoints_front[42,:], keypoints_front[25,:], keypoints_front[26,:])
	if 46 in keypoints_front_zeros or 63 in keypoints_front_zeros or 47 in keypoints_front_zeros:
		angle_wristtwr = 0
	else:
		angle_wristtwr = line_angle(keypoints_front[46,:], keypoints_front[63,:], keypoints_front[46,:], keypoints_front[47,:])
	angle_wristtw = max(abs(angle_wristtwr), abs(angle_wristtwl))

	#step9a neck twist or side bent
	if 0 in keypoints_front_zeros:
		angle_necktw = 0
	else:
		angle_necktw = line_angle(keypoints_front[0,:], keypoints_front[1,:], keypoints_front[1,:], keypoints_front[8,:])

	#step10 trunk side bend
	angle_trunkb = abs(180 - line_angle(keypoints_front[8,:],keypoints_front[1,:], keypoints_front[1,:], keypoints_front[1,:]+[0, 1]))

	#step10a trunk twist
	angle_trunktw = line_angle(keypoints_front[2,:], keypoints_front[8,:], keypoints_front[5,:], keypoints_front[8,:])



	#scoring
	#step1 &1a

	if abs(angle_uarm)<20:
		U = U+1
	elif angle_uarm>20 and angle_uarm<60:
		U = U+2
	elif angle_uarm>60:
		U = U+3
	elif angle_uarm < -20:
		U = U+2

		

	if angle_shou_abductr > 60 or angle_shou_abductl >60:
		U = U+1

	if angle_shoulder > 90 + 25:
		U = U+1


		#step2 & 2a
	if angle_larm>60 and angle_larm<100:
		L = L+1
	elif angle_larm>100 or angle_larm<60 and angle_larm>=0:
		L = L+2
	

	if abs(angle_larm_outr)>15 or abs(angle_larm_outl)>15:
		L =  L+1
	if L > 3:
		L = 3

		#step3 & 3a
	if abs(angle_wrist)<15:
		W1 = W1+1
	elif abs(angle_wrist)<45:
		W1 = W1+2
	elif abs(angle_wrist)>45:
		W1 = W1+3

	if abs(angle_wristb)>10:
		W1 = W1+1

		#step4
	if angle_wristtw <20:		#20 is angle value when wrist is not twist, maybe varies
		W2 =  W2+1
	else:
		W2 = W2+2

		#step9 & 9a
	if angle_neck>=0 and angle_neck<10:
		N = N+1
	elif angle_neck>10 and angle_neck<20:
		N = N+2
	elif angle_neck>20:
		N =N+3
	elif angle_neck<-5:
		N = N+4

	if abs(angle_necktw)>10:
		N = N+1

		#step10 & 10a
	if angle_trunk>0 and angle_trunk<20:
		T = T+1
	elif angle_trunk>20 and angle_trunk<40:
		T = T+2
	elif angle_trunk>40 and angle_trunk<60:
		T = T+3
	elif angle_trunk>60:
		T = T+4




	if angle_trunktw<30:	#30 is angle value when trunk is not twist, maybe varies
		T =  T+1

	if angle_trunkb > 10:
		T = T+1

		#step11 
	if angle_legr>170 and angle_legl>170:
		LE = LE+1
	elif angle_legr == 0 and angle_legl == 0:
		LE = 1
	else: LE = LE+2
	#lookup table A
	UL = U*10 + L
	WW = W1*100 + W2*10
	N = N*10
	TLE = T*10 + LE

	if load[0,0] > 4.4 and load[0,0] < 22:
		F1 = 1
	elif load[0,0] > 22:
		F1 = 3
	if load[0,1] > 4.4 and load[0,1] < 22:
		F2 = 1
	elif load[0,1] > 22:
		F2 = 3

	risklevel = lookup(UL,WW,N,TLE,F1,F2)
	
	angles = np.asarray([angle_uarmr,angle_uarml, angle_larmr,angle_larml,angle_shoulderr,angle_shoulderl,angle_wristr,angle_wristl, angle_larm_outr,angle_larm_outl,angle_wristbr,angle_wristbl,angle_wristtw,angle_neck, angle_necktw,angle_trunk,angle_trunkb,angle_trunktw, angle_legr, angle_legl,	U,L ,W1 ,W2 ,N ,T ,LE ,UL ,WW ,N ,TLE ,F1 ,F2])
	main_angles= np.asarray([angle_uarm, angle_larm,angle_neck, angle_trunk])
	#print(risklevel, main_angles, angles)
	return risklevel, main_angles, angles





