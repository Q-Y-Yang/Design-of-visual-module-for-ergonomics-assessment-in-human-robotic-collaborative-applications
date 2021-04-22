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





def scoring(keypoints):


	keypoints = np.delete(keypoints, -1, axis=1)
	keypoints_side_zeros = np.array(np.where(keypoints[:,0]==0.0))
		#check main keypoints coordinates
	if 5 in keypoints_side_zeros or 6 in keypoints_side_zeros or 7 in keypoints_side_zeros or 1 in keypoints_side_zeros or 8 in keypoints_side_zeros :
		return 0, 0		#if one of main keypoints is missing, stop ergonomic assessment
 
	upperarm = line_angle(keypoints[5,:],keypoints[6,:],keypoints[1,:],keypoints[8,:])
	lowerarm = line_angle(keypoints[6,:],keypoints[7,:],keypoints[1,:],keypoints[8,:])
	

	return upperarm, lowerarm

