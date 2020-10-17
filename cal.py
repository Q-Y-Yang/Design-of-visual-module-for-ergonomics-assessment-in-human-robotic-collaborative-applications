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
	angle = abs(arctan * 180 / 3.1415926)
	return angle

def main():
	#coordinates
	A1 = np.array([0,0])
	A2 = np.array([2,0])
	B1 = np.array([0,0])
	B2 =np.array([1,1.732])

	#result
	a =  line_angle(A1, A2, B1, B2)
	print(A1[2])

if __name__ == '__main__':
    main()
