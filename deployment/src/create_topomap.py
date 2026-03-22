import argparse
import os
from utils import msg_to_pil, find_images
import time

# ROS
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from message_filters import TimeSynchronizer, Subscriber
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates

from topic_names import (IMAGE_TOPIC,
                        POS_TOPIC,)
TOPOMAP_IMAGES_DIR = "../topomaps/images"
obs_img = None
world_pos = None


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def callback_obs(msg: Image):
    global obs_img
    obs_img = msg_to_pil(msg)


def callback_syn(img, pos):
    global obs_img, world_pos
    obs_img = msg_to_pil(img)
    world_pos = pos

def call_back_pos(msg: ModelStates):
    global world_pos
    world_pos = msg

def callback_joy(msg: Joy):
    if msg.buttons[0]:
        rospy.signal_shutdown("shutdown")


def main(args: argparse.Namespace):
    global obs_img, world_pos
    rospy.init_node("CREATE_TOPOMAP", anonymous=False)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    pos_curr_msg = rospy.Subscriber(
        POS_TOPIC, ModelStates, call_back_pos, queue_size=1)

    topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    if not os.path.isdir(topomap_name_dir):
        os.makedirs(topomap_name_dir)
    else:
        print(f"{topomap_name_dir} already exists. Removing previous images...")
        remove_files_in_dir(topomap_name_dir)

    assert args.dt > 0, "dt must be positive"
    rate = rospy.Rate(1/args.dt)
    print("Registered with master node. Waiting for images...")
    i = 0
    if args.pos:
        pos_list = []
    start_time = float("inf")
    while not rospy.is_shutdown():
        if obs_img is not None:
            obs_img.save(os.path.join(topomap_name_dir, f"{i}.png"))
            if args.pos and world_pos is not None:
                index = world_pos.name.index('jackal')
                pos_list.append([world_pos.pose[index].position.x, world_pos.pose[index].position.y, world_pos.pose[index].position.z, \
                                 world_pos.pose[index].orientation.x, world_pos.pose[index].orientation.y, world_pos.pose[index].orientation.z, world_pos.pose[index].orientation.w])
            print("published image", i)
            i += 1
            rate.sleep()
            start_time = time.time()
            obs_img = None
        if time.time() - start_time > 2 * args.dt:
            print(f"Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...")
            rospy.signal_shutdown("shutdown")
    if args.pos:
        print("world position is processing!")
        filename = os.path.join(topomap_name_dir, 'position.txt')
        with open(filename, 'w') as f:
            for pos in pos_list:
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {pos[3]} {pos[4]} {pos[5]} {pos[6]}\n")
        print(f"Position data saved to {filename} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 3.0)",
    )
    parser.add_argument(
        "--seg",
        "-s",
        default=False,
        type=bool,
        help=f"segmentation flag",
    )
    parser.add_argument(
        "--pos",
        "-p",
        default=True,
        type=bool,
        help=f"pos flag",
    )
    args = parser.parse_args()

    main(args)
