# ROS
from sensor_msgs.msg import Image

import numpy as np
from PIL import Image as PILImage
import cv2


def msg_to_pil(msg: Image) -> PILImage.Image:
    encoding = getattr(msg, "encoding", "rgb8").lower()
    channels = max(int(getattr(msg, "step", 0) / msg.width), 1) if msg.width else 3
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)

    if encoding in {"rgb8", "8uc3"}:
        rgb = img[:, :, :3]
    elif encoding == "bgr8":
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    elif encoding == "rgba8":
        rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif encoding == "bgra8":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif encoding in {"mono8", "8uc1"}:
        rgb = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif encoding in {"yuyv", "yuyv422", "yuv422"}:
        rgb = cv2.cvtColor(img[:, :, :2], cv2.COLOR_YUV2RGB_YUY2)
    else:
        rgb = img[:, :, :3] if channels >= 3 else cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)

    return PILImage.fromarray(rgb).convert("RGB")

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
