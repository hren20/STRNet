import os
import rospy
import torch
import argparse
import yaml
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped

from inference_utils import MODEL_REGISTRY
from inference_utils.common import load_config, inference_config_init, msg_to_pil, rotate_point_by_quaternion, create_marker_from_points
from topic_names import IMAGE_TOPIC, POS_TOPIC, REACHED_GOAL_TOPIC

# ========== 全局变量 ==========
context_queue = []
context_size = None
robo_pos = None
robo_orientation = None
rela_pos = None
closest_node = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_from_script(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))

# ========== 加载机器人参数 ==========
ROBOT_CONFIG_PATH = _resolve_from_script("../config/robot.yaml")
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# ========== ROS 回调函数 ==========
def image_callback(msg):
    img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) >= context_size + 1:
            context_queue.pop(0)
        context_queue.append(img)

def pos_callback(msg):
    global robo_pos, robo_orientation
    robo_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    robo_orientation = np.array([
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w
    ])

# ========== 主函数入口 ==========
def main(args):
    global context_size, rela_pos, closest_node

    # ===== 加载配置与模型 =====
    config, ckpt_path = load_config(args.model, _resolve_from_script(args.config))
    config = inference_config_init(config, args)
    context_size = config["context_size"]

    TrainerCls = MODEL_REGISTRY[config["model_type"]]
    trainer = TrainerCls(config=config, checkpoint_path=ckpt_path)

    # ===== 加载topomap图像与位置坐标 =====
    topomap_root = _resolve_from_script(args.topomap_root)
    topomap_dir = os.path.join(topomap_root, args.dir)
    if not os.path.isdir(topomap_dir):
        raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")
    topomap_images = sorted([
        f for f in os.listdir(topomap_dir)
        if f.endswith(".png") and f.split(".")[0].isdigit()
    ], key=lambda x: int(x.split(".")[0]))
    topomap_paths = [os.path.join(topomap_dir, name) for name in topomap_images]
    if not topomap_paths:
        raise FileNotFoundError(f"No numeric PNG topomap images found in {topomap_dir}")

    if args.pos_goal:
        position_file = os.path.join(topomap_dir, "position.txt")
        if not os.path.exists(position_file):
            raise FileNotFoundError(f"--pos-goal requires position file: {position_file}")
        positions = np.loadtxt(position_file)
        positions = np.atleast_2d(positions)
        if len(positions) != len(topomap_paths):
            raise ValueError(
                f"position.txt rows ({len(positions)}) must match topomap images ({len(topomap_paths)})"
            )
    else:
        positions = None

    closest_node = args.init_node
    goal_node = len(topomap_paths) - 1 if args.goal_node == -1 else args.goal_node
    if not 0 <= closest_node < len(topomap_paths):
        raise ValueError(f"--init-node must be between 0 and {len(topomap_paths) - 1}, got {closest_node}")
    if not 0 <= goal_node < len(topomap_paths):
        raise ValueError(f"--goal-node must be -1 or between 0 and {len(topomap_paths) - 1}, got {goal_node}")

    # ===== ROS 初始化 =====
    rospy.init_node("navigate_node", anonymous=False)
    rospy.Subscriber(args.image_topic, Image, image_callback, queue_size=1)
    if args.pos_goal:
        rospy.Subscriber(args.pos_topic, PoseStamped, pos_callback, queue_size=1)

    waypoint_pub = rospy.Publisher(args.waypoint_topic, Float32MultiArray, queue_size=1)
    subgoal_marker_pub = rospy.Publisher(args.subgoal_marker_topic, Marker, queue_size=1)
    goal_marker_pub = rospy.Publisher(args.goal_marker_topic, Marker, queue_size=1)
    goal_status_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1)

    # ===== 额外可视化 Publisher =====
    marker_pub = rospy.Publisher(args.sampled_marker_topic, Marker, queue_size=10)
    sampled_actions_pub = rospy.Publisher(args.sampled_actions_topic, Float32MultiArray, queue_size=1)


    rate = rospy.Rate(config.get("frame_rate", RATE))

    scale = 3
    scale_factor = scale * MAX_V / RATE

    while not rospy.is_shutdown():
        if len(context_queue) < context_size + 1:
            rate.sleep()
            continue

        obs_tensor = trainer.prepare_inputs(context_queue)

        # ===== 子目标选择 =====
        if args.pos_goal and (robo_pos is None or robo_orientation is None):
            rate.sleep()
            continue
        using_pos_goal = args.pos_goal
        if using_pos_goal:
            distances = np.linalg.norm(positions[:, :2] - robo_pos[:2], axis=1)
            min_idx = int(np.argmin(distances))
            rela_pos = rotate_point_by_quaternion(positions[min_idx][:3] - robo_pos, robo_orientation)[:2]
            goal_tensor = trainer.prepare_inputs([topomap_paths[min_idx]])
            closest_node = min_idx
            goal_tensors = goal_tensor
            obs_tensor = obs_tensor.repeat(goal_tensors.shape[0], 1, 1, 1)
        else:
            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            subset_paths = topomap_paths[start:end + 1]
            if not subset_paths:
                raise RuntimeError(
                    f"Empty topomap window: start={start}, end={end}, closest_node={closest_node}, goal_node={goal_node}"
                )
            goal_tensors = torch.cat([trainer.prepare_inputs([p]) for p in subset_paths], dim=0)
            obs_tensor = obs_tensor.repeat(goal_tensors.shape[0], 1, 1, 1)

        # ===== 推理动作与距离预测 =====
        actions, min_idx = trainer.toponavi_policy(
            obs_tensor,
            goal_images=goal_tensors,
            num_samples=args.num_samples
        )

        if not using_pos_goal:
            closest_node = start + int(min_idx)
        print("closest_node: ", closest_node)
        if not 0 <= args.waypoint < len(actions[0]):
            raise ValueError(f"--waypoint must be between 0 and {len(actions[0]) - 1}, got {args.waypoint}")
        chosen_waypoint = actions[0][args.waypoint]
        if config.get("normalize", False):
            chosen_waypoint[:2] *= (scale_factor / scale)

        msg = Float32MultiArray()
        msg.data = chosen_waypoint.astype(np.float32).tolist()
        waypoint_pub.publish(msg)

        # ====== 发布采样动作可视化 ======
        sampled_actions_msg = Float32MultiArray()
        action_sample = np.asarray(actions[0])
        flat_action = action_sample.flatten()         # 取第 1 条采样轨迹
        sampled_actions_msg.data = np.concatenate(([0], flat_action)).astype(np.float32).tolist()

        traj_pts = action_sample[:, :2] * scale_factor

        marker = create_marker_from_points(
            traj_pts,
            color=(1.0, 0.0, 0.0),   # 红色
            scale=0.08,              # 点大小
            frame_id="base_link",
            z_value=0.0,
            marker_id=0,
            namespace="sampled_action_traj"
        )
        marker_pub.publish(marker)
        sampled_actions_pub.publish(sampled_actions_msg)

        if using_pos_goal and rela_pos is not None:
            goal_marker = create_marker_from_points([rela_pos], color=(0, 1, 0))
            goal_marker_pub.publish(goal_marker)

        # ===== 判断是否到达终点 =====
        reached_goal = closest_node >= goal_node
        goal_status_pub.publish(Bool(data=reached_goal))
        if reached_goal:
            rospy.loginfo("Reached goal! Halting.")
            while not rospy.is_shutdown():
                goal_status_pub.publish(Bool(data=True))
                rate.sleep()
        rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="strnetnew")
    parser.add_argument("--config", type=str, default="../config/models.yaml")
    parser.add_argument("--topomap-root", type=str, default="../topomaps/images")
    parser.add_argument("--dir", type=str, default="collision_forward")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--waypoint", type=int, default=2)
    parser.add_argument("--pos-goal", action="store_true", help="是否使用位置目标")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--init-node", type=int, default=0, help="导航起始节点")
    parser.add_argument("--goal-node", type=int, default=40, help="目标节点 (-1 表示最后一个节点)")
    parser.add_argument("--image-topic", type=str, default=IMAGE_TOPIC)
    parser.add_argument("--pos-topic", type=str, default=POS_TOPIC)
    parser.add_argument("--waypoint-topic", type=str, default="/waypoint")
    parser.add_argument("--subgoal-marker-topic", type=str, default="/goal")
    parser.add_argument("--goal-marker-topic", type=str, default="/topoplan/goal_marker")
    parser.add_argument("--close-threshold","-t",default=3,type=int,
                        help="""temporal distance within the next node in the topomap before localizing to it (default: 3)""",)
    parser.add_argument("--sampled-marker-topic", type=str,
                        default="/path")
    parser.add_argument("--sampled-actions-topic", type=str,
                        default="/sampled_actions")
    args = parser.parse_args()
    main(args)
