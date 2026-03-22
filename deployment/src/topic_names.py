# topic names for ROS communication

# 替换 topic_names.py 或直接在脚本中设置
IMAGE_TOPIC = "/carla/ego_vehicle/rgb_front/image"
POS_TOPIC = "/carla/ego_vehicle/odometry"


# exploration topics
WAYPOINT_TOPIC = "/waypoint"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"

# recorded ont the robot
ODOM_TOPIC = "/odom"
BUMPER_TOPIC = "/mobile_base/events/bumper"
JOY_BUMPER_TOPIC = "/joy_bumper"

# move the robot