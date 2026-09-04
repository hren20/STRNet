import os
import yaml

# ROS
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from topic_names import JOY_BUMPER_TOPIC

vel_msg = Twist()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../config/robot.yaml"))
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_teleop_topic"]
JOY_CONFIG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../config/joystick.yaml"))
with open(JOY_CONFIG_PATH, "r") as f:
	joy_config = yaml.safe_load(f)
DEADMAN_SWITCH = joy_config["deadman_switch"] # button index
LIN_VEL_BUTTON = joy_config["lin_vel_button"]
ANG_VEL_BUTTON = joy_config["ang_vel_button"]
RATE = 30
vel_pub = None
bumper_pub = None
button = None
bumper = False


def _get_index(values, index: int, default=0):
	return values[index] if 0 <= index < len(values) else default


def callback_joy(data: Joy):
	"""Callback function for the joystick subscriber"""
	global vel_msg, button, bumper
	button = bool(_get_index(data.buttons, DEADMAN_SWITCH, 0))
	bumper_button = bool(_get_index(data.buttons, DEADMAN_SWITCH - 1, 0))
	if button: # hold down the dead-man switch to teleop the robot
		vel_msg.linear.x = MAX_V * _get_index(data.axes, LIN_VEL_BUTTON, 0.0)
		vel_msg.angular.z = MAX_W * _get_index(data.axes, ANG_VEL_BUTTON, 0.0)
	else:
		vel_msg = Twist()
		if vel_pub is not None:
			vel_pub.publish(vel_msg)
	bumper = bumper_button



def main():
	global vel_pub, bumper_pub
	rospy.init_node("Joy2Locobot", anonymous=False)
	vel_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	bumper_pub = rospy.Publisher(JOY_BUMPER_TOPIC, Bool, queue_size=1)
	joy_sub = rospy.Subscriber("joy", Joy, callback_joy)
	rate = rospy.Rate(RATE)	
	print("Registered with master node. Waiting for joystick input...")
	while not rospy.is_shutdown():
		if button:
			print(f"Teleoperating the robot:\n {vel_msg}")
			vel_pub.publish(vel_msg)
		bumper_msg = Bool()
		bumper_msg.data = bumper
		bumper_pub.publish(bumper_msg)
		if bumper:
			print("Bumper pressed!")
		rate.sleep()


if __name__ == "__main__":
	main()
