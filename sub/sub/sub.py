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
            'keypoints',
            self.listener_callback,
            10)
		self.subscription  # prevent unused variable warning


	def listener_callback(self, msg):
		result2txt = str(msg.data)
		with open('/home/student/pose/keypoints4.txt', 'a') as file_handle:
			file_handle.write(result2txt)
			file_handle.write('\n')

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
