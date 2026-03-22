import yaml, torch
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import rospy
import numpy as np
from typing import List, Sequence, Tuple

from torchvision import transforms
import torchvision.transforms.functional as TF

# models
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO


def create_marker_from_points(
    points: Sequence[np.ndarray],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    scale: float = 0.1,
    frame_id: str = "base_link",
    z_value: float = 0.0,
    marker_id: int = 0,
    namespace: str = "points",
    enforce_eight_points: bool = True,
) -> Marker:
    """
    创建用于 RViz 显示的 Marker 点云（POINTS 类型）

    Args:
        points (Sequence[np.ndarray]): 2D 坐标列表，每个为 (x, y)
        color (Tuple[float, float, float]): RGB 值 (0-1)，默认红色
        scale (float): 每个点的尺寸
        frame_id (str): ROS 坐标系 ID
        z_value (float): 所有点的 Z 坐标高度
        marker_id (int): marker 的唯一标识
        namespace (str): marker 的命名空间

    Returns:
        Marker: 可发布的可视化消息
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    marker.scale.x = scale
    marker.scale.y = scale
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]

    # Pad with zeros if enforce_eight_points=True and fewer than 8 points
    if enforce_eight_points and len(points) < 8:
        padding = np.zeros((8 - len(points), 2))  # Zero-pad to 8 points
        points = np.vstack([points, padding])

    for pt in points:
        pt = np.asarray(pt)
        p = Point()
        p.x = float(pt[0])
        p.y = float(pt[1])
        p.z = z_value
        marker.points.append(p)

    return marker

def rotate_point_by_quaternion(point, quaternion):
    # 使用 scipy 直接进行四元数到旋转矩阵的转换
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()

    # 将点转换为3x1向量
    point_h = np.array([point[0], point[1], point[2]])

    # 应用旋转矩阵
    rotated_point = np.dot(np.linalg.inv(rotation_matrix), point_h)

    return rotated_point

def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image

def load_config(model_key: str, config_path: str):
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    model_info = full_config[model_key]
    with open(model_info["config_path"], "r") as f:
        model_config = yaml.safe_load(f)
    return model_config, model_info["ckpt_path"]

def load_images(context_size, folder_path, image_size) -> List[str]:
    return [f"{folder_path}/{i}.png" for i in range(context_size)]

def inference_config_init(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    config["train"] = False
    config["num_samples"] = args.num_samples
    config["close_threshold"] = args.close_threshold
    return config

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)