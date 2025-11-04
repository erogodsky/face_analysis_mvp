import math

import numpy as np
import cv2


def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

def bbox_trbl2xywh(bbox):
    """
    Convert bounding box from (top, right, bottom, left) format
    to (x, y, w, h) format used by OpenCV and many CV libraries.

    Args:
        bbox (tuple or list): (top, right, bottom, left)

    Returns:
        tuple: (x, y, w, h)
    """
    top, right, bottom, left = bbox
    x = left
    y = top
    w = right - left
    h = bottom - top
    return np.array([x, y, w, h])

def pad_resize_image(img, dst_size):
    """
    Resize the image so that the largest side becomes dst_size while preserving aspect ratio,
    then pad the image to 'dst_size'x'dst_size', centering it. Padding color is (127, 127, 127).

    Args:
        img (np.ndarray): Input image (H, W, C)
        dst_size (int): destination size

    Returns:
        np.ndarray: Padded 'dst_size'x'dst_size' image
        tuple: (scale, top, left) — scaling factor and padding offsets
    """
    h, w = img.shape[:2]
    scale = dst_size / max(h, w)

    # Resize keeping aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding to center the image
    pad_top = (256 - new_h) // 2
    pad_bottom = 256 - new_h - pad_top
    pad_left = (256 - new_w) // 2
    pad_right = 256 - new_w - pad_left

    # Apply padding
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(127, 127, 127)
    )

    return padded, (scale, pad_top, pad_left)
