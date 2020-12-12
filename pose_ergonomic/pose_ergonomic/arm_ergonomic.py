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
		return 0





def scoring(keypoints):


	keypoints = np.delete(keypoints, -1, axis=1)

	#check main keypoints coordinates
	if keypoints[5,:] == [0,0] or keypoints[6,:] == [0,0] or keypoints[7,:] == [0,0] or keypoints[1,:] == [0,0] or keypoints[8,:] == [0,0]:
		return 0, 0, 0			#if one of main keypoints is missing, stop ergonomic assessment
 
	upperarm = line_angle(keypoints[5,:],keypoints[6,:],keypoints[1,:],keypoints[8,:])
	lowerarm = line_angle(keypoints[5,:],keypoints[6,:],keypoints[6,:],keypoints[7,:])
	

	return upperarm, lowerarm

