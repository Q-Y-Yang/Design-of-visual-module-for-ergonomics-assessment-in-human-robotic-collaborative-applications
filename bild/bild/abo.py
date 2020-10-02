#!/usr/bin/env python
#!coding=utf-8
 
#right code !
#function: 
#display the frame from another node.
 
import rclpy
from rclpy.node import Node
#import numpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class Abo(Node):
    def __init__(self):
        super().__init__('abo')
        self.br = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.callback,
            10)
        self.subscription  # prevent unused variable warning

 
    def callback(self,data):
    # define picture to_down' coefficient of ratio
        cv_img = self.br.imgmsg_to_cv2(data, "bgr8")
        cv2.imshow("frame" , cv_img)
        cv2.waitKey(3)
       
def main(args=None):
    rclpy.init(args=args)

    abo = Abo()

    rclpy.spin(abo)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    abo.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()

