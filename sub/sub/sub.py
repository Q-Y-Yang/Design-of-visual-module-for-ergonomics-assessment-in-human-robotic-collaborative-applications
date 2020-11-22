import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray


class Subscriber(Node):

	def __init__(self):
		super().__init__('subscriber')
		time = Header()
		time.stamp = Node.get_clock(self).now().to_msg()
		self.subscription = self.create_subscription(
            Float32MultiArray,
            'keypoints_side',
            self.listener_callback,
            10)
		self.subscription = self.create_subscription(
            Float32MultiArray,
            'keypoints_front',
            self.listener_callback,
            10)
		self.subscription  # prevent unused variable warning


	def listener_callback(self, msg):
		#result2txt = str(msg.data)
		#with open('/home/student/pose/keypoints4.txt', 'a') as file_handle:
		#	file_handle.write(result2txt)
		#	file_handle.write('\n')
			# read keypoints
		keypointsarr = np.array(msg.data).reshape(46,3)	 #2D array for a person
		keypoints_side = np.delete(keypointsarr, -1, axis=1)   #delete confidence score
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
		angle11 = line_angle(keypoints_side[2,:], keypoints_side[3,:], keypoints_side[8,:], keypoints_side[1,:])  #normally from side view both shoulders can be detected.
		#angle12 = line_angle(keypoints[5], keypoints[6], keypoints[8], keypoints[1])

		#step1a shoulder raised?
		angle13 = line_angle(keypoints[2,:], keypoints_side[1,:],keypoints_side[8,:], keypoints_side[1,:])  

		#step2 lower arm
		angle21 = line_angle(keypoints_side[3,:], keypoints_side[4,:], keypoints_side[2,:], keypoints_side[3,:])

		#step3 wrist  also need hand points
		angle31 = line_angle(keypoints_side[3,:], keypoints_side[4,:], keypoints_side[4,:], keypoints_side[34,:])
		#step4 wrist twist
		angle41 = line_angle(keypoints_side[25,:], keypoints_side[34,:], keypoints_side[25,:], keypoints_side[27,:])
		#step9 neck
		angle91 = line_angle(keypoints_side[8,:], keypoints_side[1,:], keypoints_side[1,:], keypoints_side[17,:])

		#step10 trunk
		angle101 = abs(line_angle(keypoints_side[8,:],keypoints_side[1,:], keypoints_side[1,:], keypoints_side[1,:]+[1, 0]) - 90)

		#from top front view
		#step1a abduction of shoulder
		angle1_1 = line_angle(keypoints_front[2,:], keypoints_front[3,:], keypoints_front[8,:], keypoints_front[1,:])
		angle1_2 = line_angle(keypoints_front[5,:], keypoints_front[6,:], keypoints_front[8,:], keypoints_front[1,:])

		#step2a lower arm outside of body
		angle2_1 = line_angle(keypoints_front[3,:], keypoints_front[4,:], keypoints_front[8,:], keypoints_front[1,:])
		angle2_2 = line_angle(keypoints_front[6,:], keypoints_front[7,:], keypoints_front[8,:], keypoints_front[1,:])

		#step3a wrist bent from midline
		angle3_1 = line_angle(keypoints_front[3,:], keypoints_front[4,:],keypoints_front[4,:], keypoints_front[34,:])

		#step9a neck twist or side bent
		angle9_1 = line_angle(keypoints_front[0,:], keypoints_front[1,:], keypoints_front[8,:], keypoints_front[1,:])

		#step10 trunk side bend
		angle10_1 = abs(line_angle(keypoints_front[8,:],keypoints_front[1,:], keypoints_front[1,:], keypoints_front[1,:]+[1, 0]) - 90)

		#step10a trunk twist
		angle10_2 = line_angle(keypoints_front[2,:], keypoints_front[8,:], keypoints_front[5,:], keypoints_front[8,:])

		#step11 legs evenly-balanced
		angle11_1 = line_angle(keypoints_front[9,:], keypoints_front[10,:], keypoints_front[10,:], keypoints_front[11,:])
		angle11_2 = line_angle(keypoints_front[12,:], keypoints_front[13,:], keypoints_front[13,:], keypoints_front[14,:])


		#scoring
		#step1 &1a
		if abs(angle11)<20:
			U = U+1
		elif angle11<-20 or angle11>20 and angle11<45:
			U = U+2
		elif angle11>45 and angle11<90:
			U = U+3
		elif angle11>90:
			U = U+4

		if angle1_1 > 45 or angle1_2 >45:
			U = U+1

		if angle13 > 90:
			U = U+1

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

		if angle10_1 < 40:
			T = T+1

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


def main(args=None):
	rclpy.init(args=args)

	subscriber = Subscriber()

	rclpy.spin(subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
	subscriber.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
