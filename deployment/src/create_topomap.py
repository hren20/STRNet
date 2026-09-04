import argparse
import os
import shutil
import time
from utils import msg_to_pil

# ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

from topic_names import (IMAGE_TOPIC,
                        POS_TOPIC,)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPOMAP_IMAGES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../topomaps/images"))
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

def call_back_pos(msg):
    global world_pos
    world_pos = msg


def pose_to_list(pose):
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]


def extract_position_sample(msg, args):
    if args.pos_type == "pose_stamped":
        return pose_to_list(msg.pose)

    try:
        index = msg.name.index(args.robot_name)
    except ValueError:
        rospy.logwarn_throttle(
            5.0,
            f"Robot '{args.robot_name}' not found in {args.pos_topic}; skipping position sample.",
        )
        return None
    return pose_to_list(msg.pose[index])


def main(args: argparse.Namespace):
    global obs_img, world_pos
    rospy.init_node("CREATE_TOPOMAP", anonymous=False)
    image_curr_msg = rospy.Subscriber(
        args.image_topic, Image, callback_obs, queue_size=1)
    if args.pos:
        if args.pos_type == "model_states":
            from gazebo_msgs.msg import ModelStates
            pos_msg_type = ModelStates
        else:
            pos_msg_type = PoseStamped
        pos_curr_msg = rospy.Subscriber(
            args.pos_topic, pos_msg_type, call_back_pos, queue_size=1)

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
            pos_sample = None
            if args.pos:
                if world_pos is None:
                    rospy.logwarn_throttle(
                        5.0,
                        f"Waiting for pose samples from {args.pos_topic} before saving topomap frames.",
                    )
                    rate.sleep()
                    continue
                pos_sample = extract_position_sample(world_pos, args)
                if pos_sample is not None:
                    pos_list.append(pos_sample)
                else:
                    rate.sleep()
                    continue
            obs_img.save(os.path.join(topomap_name_dir, f"{i}.png"))
            print("published image", i)
            i += 1
            rate.sleep()
            start_time = time.time()
            obs_img = None
        else:
            rate.sleep()
        if time.time() - start_time > 2 * args.dt:
            print(f"Topic {args.image_topic} not publishing anymore. Shutting down...")
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
        description=f"Code to generate topomaps from an image topic (default: {IMAGE_TOPIC})"
    )
    parser.add_argument(
        "--image-topic",
        default=IMAGE_TOPIC,
        type=str,
        help=f"image topic to sample (default: {IMAGE_TOPIC})",
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
        help=f"time between sampled images (default: 1.0)",
    )
    parser.add_argument(
        "--seg",
        "-s",
        action="store_true",
        help=f"segmentation flag",
    )
    parser.add_argument(
        "--pos",
        "-p",
        dest="pos",
        action="store_true",
        help=f"save robot poses from a pose topic (default topic: {POS_TOPIC})",
    )
    parser.add_argument(
        "--no-pos",
        dest="pos",
        action="store_false",
        help="do not save robot poses",
    )
    parser.set_defaults(pos=False)
    parser.add_argument(
        "--pos-topic",
        default=POS_TOPIC,
        type=str,
        help=f"pose topic to save with --pos (default: {POS_TOPIC})",
    )
    parser.add_argument(
        "--pos-type",
        choices=("pose_stamped", "model_states"),
        default="pose_stamped",
        help="message type for --pos-topic (default: pose_stamped)",
    )
    parser.add_argument(
        "--robot-name",
        default="jackal",
        type=str,
        help="robot model name when --pos-type model_states is used (default: jackal)",
    )
    args = parser.parse_args()

    main(args)
