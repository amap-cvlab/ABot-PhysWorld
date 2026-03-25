"""
Action 控制视频生成工具模块

移植自 Genie-Envisioner 仓库，用于 TI-Action-2V 的训练和推理。

核心功能：
1. 轨迹图生成 (get_traj_maps): 将 action 的空间信息可视化为图像
2. 射线图生成 (get_ray_maps): 生成相机射线信息（射线原点和方向）
3. Action 数据加载: 从 HDF5 文件加载 action 数据

参考文档: /mnt/workspace/gaowo/LongLive/Genie-Envisioner/ACTION_CONTROL_VIDEO_GENERATION_ANALYSIS.md
"""

import numpy as np
import cv2
import matplotlib.cm as cm
import torch
from einops import rearrange
import h5py
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, List, Union


# ============== 四元数和变换矩阵工具 ==============

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Copied from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L43C1-L72C54

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_transformation_matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    从四元数和位置构建变换矩阵
    
    Args:
        quat: shape (b, 7)，前3个是位置xyz，后4个是四元数xyzw
    
    Returns:
        变换矩阵 shape (b, 4, 4)
    """
    rot_quat = quat[:, 3:]
    rot_quat = rot_quat[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    rot = quaternion_to_matrix(rot_quat)
    trans = quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(quat.shape[0], 1, 1).to(quat.device, quat.dtype)
    output[:, :3, :3] = rot
    output[:, :3, 3] = trans
    return output


# ============== 轨迹图生成 ==============

def simple_radius_gen_func(xyzs: torch.Tensor, c_xyzs: torch.Tensor) -> torch.Tensor:
    """
    A simple empirical function to generate radius based on the distances 
    between end-effectors and the camera
    
    Args:
        xyzs: 末端执行器位置 (t, 3)
        c_xyzs: 相机位置 (t, 3)
    
    Returns:
        半径值 (t,)
    """
    radius = torch.clamp(
        1.0 - torch.sqrt(((xyzs - c_xyzs) ** 2).sum(-1)) - 0.07 / (0.8 - 0.07), 
        min=0, 
        max=1
    ) * 100
    return radius


def get_traj_maps(
    pose: Union[np.ndarray, torch.Tensor], 
    w2c: torch.Tensor, 
    c2w: torch.Tensor, 
    intrinsic: torch.Tensor, 
    sample_size: Tuple[int, int], 
    radius_gen_func=None
) -> torch.Tensor:
    """
    生成轨迹图（Pose2Image Conditioning）
    
    将 action 的空间信息可视化为图像：
    1. 3D 位置投影到 2D 像素坐标
    2. 姿态可视化（旋转矩阵的三个正交轴向量投影为箭头）
    3. 夹爪开合度可视化（用颜色深浅表示）
    4. 左右臂区分（左臂绿色系，右臂红色系）
    
    Args:
        pose: action 数据, shape (t, c)，其中 c 通常为 16
              格式: [left_xyz(3), left_quat(4), left_gripper(1), right_xyz(3), right_quat(4), right_gripper(1)]
        w2c: world-to-camera 变换矩阵, shape (v, t, 4, 4)
        c2w: camera-to-world 变换矩阵, shape (v, t, 4, 4)
        intrinsic: 相机内参矩阵, shape (v, 3, 3)
        sample_size: 输出图像尺寸 (h, w)
        radius_gen_func: 可选的半径生成函数
    
    Returns:
        轨迹图, shape (c, v, t, h, w)，其中 c=3 (RGB)
    """
    h, w = sample_size
    colormap_l = cm.Greens
    colormap_r = cm.Reds
    color_list_l = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
    color_list_r = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]

    if isinstance(pose, np.ndarray):
        pose = torch.tensor(pose, dtype=torch.float32)
    
    device = pose.device

    ee_key_pts = torch.tensor([
        [0, 0, 0, 1],
        [0.1, 0, 0, 1],
        [0, 0.1, 0, 1],
        [0, 0, 0.1, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 4, 4).permute(0, 1, 3, 2)

    # 1, t, 4, 4
    pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(dim=0)
    pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15]).unsqueeze(dim=0)

    # v, t, 4, 4
    ee2cam_l = torch.matmul(w2c, pose_l_mat)
    ee2cam_r = torch.matmul(w2c, pose_r_mat)

    correct_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.23],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 4, 4)
    ee2cam_l = torch.matmul(ee2cam_l, correct_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, correct_matrix)

    # v, t, 4, 4
    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)

    # v, 1, 3, 3
    intrinsic = intrinsic.unsqueeze(1)

    # v, t, 3, 4
    uvs_l0 = torch.matmul(intrinsic, pts_l[:, :, :3, :])
    uvs_l = (uvs_l0 / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)

    # v, t, 3, 4
    uvs_r0 = torch.matmul(intrinsic, pts_r[:, :, :3, :])
    uvs_r = (uvs_r0 / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)

    all_img_list = []

    for icam in range(w2c.shape[0]):
        l_xyz = pose[:, 0:3].clone()
        r_xyz = pose[:, 8:11].clone()
        c_xyz = c2w[icam, :, :3, 3].clone()

        if radius_gen_func is None:
            l_dist = torch.full((pose.shape[0],), 50, device=device)
            r_dist = torch.full((pose.shape[0],), 50, device=device)
        else:
            l_dist = radius_gen_func(l_xyz, c_xyz)
            r_dist = radius_gen_func(r_xyz, c_xyz)

        img_list = []
        for i in range(pose.shape[0]):
            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            normalized_value_l = pose[i, 7].item() / 120
            normalized_value_r = pose[i, 15].item() / 120
            color_l = colormap_l(normalized_value_l)[:3]  # Get RGB values
            color_r = colormap_r(normalized_value_r)[:3]  # Get RGB values
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            for points, color, colors, radius, lr_tag, eef in zip(
                [uvs_l[icam, i], uvs_r[icam, i]], 
                [color_l, color_r], 
                [color_list_l, color_list_r], 
                [l_dist[i], r_dist[i]], 
                ["left", "right"], 
                [normalized_value_l, normalized_value_r]
            ):
                base = np.array(points[0].cpu())
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                point = np.array(points[0][:2].cpu())
                radius = int(radius.item() if torch.is_tensor(radius) else radius)
                cv2.circle(img, tuple(point), radius, color, -1)

            for points, color, colors, lr_tag in zip(
                [uvs_l[icam, i], uvs_r[icam, i]], 
                [color_l, color_r], 
                [color_list_l, color_list_r], 
                ["left", "right"]
            ):
                base = np.array(points[0].cpu())
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                for j, point in enumerate(points):
                    point = np.array(point[:2].cpu())
                    if j == 0:
                        continue
                    else:
                        cv2.line(img, tuple(base), tuple(point), colors[j - 1], 8)

            img_list.append(img / 255.)

        img_list = np.stack(img_list, axis=0)  # t, h, w, c
        all_img_list.append(img_list)

    all_img_list = np.stack(all_img_list, axis=0)  # ncam, t, h, w, c
    all_img_list = rearrange(torch.tensor(all_img_list), "v t h w c -> c v t h w").float()

    return all_img_list


# ============== 射线图生成 ==============

def get_ray_maps(
    intrinsic: torch.Tensor, 
    c2w: torch.Tensor, 
    H: int, 
    W: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成射线图（Ray Maps）
    
    生成相机射线信息（射线原点和方向），用于补充相机几何信息。
    
    Args:
        intrinsic: 相机内参矩阵, shape (vt, 3, 3)
        c2w: camera-to-world 变换矩阵, shape (vt, 4, 4)
        H: 图像高度
        W: 图像宽度
    
    Returns:
        rays_o: 射线原点, shape (vt, H, W, 3)
        viewdir: 射线方向（归一化）, shape (vt, H, W, 3)
    """
    vt = intrinsic.shape[0]
    device = c2w.device
    
    fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2)
    fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2)
    cx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2)
    cy = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2)
    
    # 注意：原始代码使用 PyTorch 默认的 'ij' indexing 模式
    i, j = torch.meshgrid(
        torch.linspace(0.5, W - 0.5, W, device=device), 
        torch.linspace(0.5, H - 0.5, H, device=device),
        indexing='ij'
    )
    i = i.t()
    j = j.t()
    i = i.unsqueeze(0).repeat(vt, 1, 1)
    j = j.unsqueeze(0).repeat(vt, 1, 1)
    
    dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3, :3], -1)
    rays_o = c2w[:, :3, -1].unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1)
    viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    return rays_o, viewdir


# ============== Action 数据加载 ==============

def normalize_angles(radius: np.ndarray) -> np.ndarray:
    """将角度归一化到 [-pi, pi] 范围"""
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
    return radius_normed


def get_actions_eef(
    gripper: np.ndarray, 
    all_ends_p: np.ndarray = None, 
    all_ends_o: np.ndarray = None, 
    slices: List[int] = None, 
    delta_act_sidx: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取末端执行器空间的 action
    
    Args:
        gripper: 夹爪位置数据
        all_ends_p: 末端执行器位置 (n, 2, 3)
        all_ends_o: 末端执行器姿态（四元数）(n, 2, 4)
        slices: 采样索引
        delta_act_sidx: delta action 起始索引
    
    Returns:
        all_abs_actions: 绝对 action, shape (n, 14)
        all_delta_actions: delta action, shape (n-delta_act_sidx, 14)
    """
    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        n = all_ends_p.shape[0] - 1 + delta_act_sidx
        slices = [0,] * (delta_act_sidx - 1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_left_rpy = []
    all_right_rpy = []

    for i in slices:
        rot_l = Rotation.from_quat(all_ends_o[i, 0])
        left_rpy = np.concatenate((all_ends_p[i, 0], rot_l.as_euler("xyz", degrees=False)), axis=0)
        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        right_rpy = np.concatenate((all_ends_p[i, 1], rot_r.as_euler("xyz", degrees=False)), axis=0)
        all_left_rpy.append(left_rpy)
        all_right_rpy.append(right_rpy)

    # xyz, rpy
    all_left_rpy = np.stack(all_left_rpy)
    all_right_rpy = np.stack(all_right_rpy)

    # xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 14])
    # xyz, rpy, gripper
    all_delta_actions = np.zeros([n - delta_act_sidx, 14])
    
    for i in range(0, n):
        all_abs_actions[i, 0:6] = all_left_rpy[i, :6]
        all_abs_actions[i, 6] = gripper[slices[i], 0]
        all_abs_actions[i, 7:13] = all_right_rpy[i, :6]
        all_abs_actions[i, 13] = gripper[slices[i], 1]
        if i >= delta_act_sidx:
            all_delta_actions[i - delta_act_sidx, 0:6] = all_left_rpy[i, :6] - all_left_rpy[i - 1, :6]
            all_delta_actions[i - delta_act_sidx, 3:6] = normalize_angles(all_delta_actions[i - delta_act_sidx, 3:6])
            all_delta_actions[i - delta_act_sidx, 6] = gripper[slices[i], 0]
            all_delta_actions[i - delta_act_sidx, 7:13] = all_right_rpy[i, :6] - all_right_rpy[i - 1, :6]
            all_delta_actions[i - delta_act_sidx, 10:13] = normalize_angles(all_delta_actions[i - delta_act_sidx, 10:13])
            all_delta_actions[i - delta_act_sidx, 13] = gripper[slices[i], 1]

    return all_abs_actions, all_delta_actions


def get_actions_joint(
    gripper: np.ndarray, 
    all_joints: np.ndarray = None, 
    slices: List[int] = None, 
    delta_act_sidx: int = None, 
    n_arm_joints: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取关节空间的 action
    
    Args:
        gripper: 夹爪位置数据
        all_joints: 关节角度 (n, n_arm_joints*2)
        slices: 采样索引
        delta_act_sidx: delta action 起始索引
        n_arm_joints: 每个臂的关节数
    
    Returns:
        all_abs_actions: 绝对 action, shape (n, n_arm_joints*2+2)
        all_delta_actions: delta action
    """
    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        n = all_joints.shape[0] - 1 + delta_act_sidx
        slices = [0,] * (delta_act_sidx - 1) + list(range(all_joints.shape[0]))
    else:
        n = len(slices)

    all_abs_actions = np.zeros([n, n_arm_joints * 2 + 2])
    all_delta_actions = np.zeros([n - delta_act_sidx, n_arm_joints * 2 + 2])
    
    for i in range(0, n):
        i_joint_l = all_joints[slices[i]][:n_arm_joints]
        i_joint_r = all_joints[slices[i]][n_arm_joints:]
        all_abs_actions[i, :n_arm_joints] = i_joint_l
        all_abs_actions[i, n_arm_joints] = gripper[slices[i], 0]
        all_abs_actions[i, n_arm_joints + 1:2 * n_arm_joints + 1] = i_joint_r
        all_abs_actions[i, 2 * n_arm_joints + 1] = gripper[slices[i], 1]
        if i >= delta_act_sidx:
            all_delta_actions[i - delta_act_sidx, :n_arm_joints] = i_joint_l - all_joints[slices[i] - 1][:n_arm_joints]
            all_delta_actions[i - delta_act_sidx, n_arm_joints] = gripper[slices[i], 0]
            all_delta_actions[i - delta_act_sidx, n_arm_joints + 1:2 * n_arm_joints + 1] = i_joint_r - all_joints[slices[i] - 1][n_arm_joints:]
            all_delta_actions[i - delta_act_sidx, 2 * n_arm_joints + 1] = gripper[slices[i], 1]

    return all_abs_actions, all_delta_actions


def parse_h5(
    h5_file: str, 
    slices: List[int] = None, 
    delta_act_sidx: int = 1, 
    action_space: str = "eef", 
    n_arm_joints: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取并解析 .h5 文件，获取绝对 action 和 action 差值
    
    Args:
        h5_file: HDF5 文件路径
        slices: 采样索引列表
        delta_act_sidx: delta action 起始索引
        action_space: "eef" (末端执行器) 或 "joint" (关节空间)
        n_arm_joints: 每个臂的关节数
    
    Returns:
        all_abs_actions: 绝对 action
        all_delta_actions: delta action
    """
    with h5py.File(h5_file, "r") as fid:
        all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)

        if action_space == "eef":
            all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
            all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)
            all_abs_actions, all_delta_actions = get_actions_eef(
                gripper=all_abs_gripper,
                slices=slices,
                delta_act_sidx=delta_act_sidx,
                all_ends_p=all_ends_p,
                all_ends_o=all_ends_o,
            )
        elif action_space == "joint":
            all_joints = np.array(fid["state/joint/position"])
            all_abs_actions, all_delta_actions = get_actions_joint(
                gripper=all_abs_gripper,
                slices=slices,
                delta_act_sidx=delta_act_sidx,
                all_joints=all_joints,
                n_arm_joints=n_arm_joints
            )
        else:
            raise NotImplementedError(f"Unknown action_space: {action_space}")

    return all_abs_actions, all_delta_actions


def load_actions_with_quat(
    h5_file: str,
    slices: Optional[List[int]] = None
) -> np.ndarray:
    """
    从 HDF5 文件加载四元数格式的 action 数据（用于轨迹图可视化）
    
    与 parse_h5 不同，此函数保留原始的四元数表示，而不是转换为欧拉角。
    这是 get_traj_maps 所需的格式。
    
    Args:
        h5_file: HDF5 文件路径
        slices: 采样索引列表
    
    Returns:
        actions: shape (n, 16)
            格式: [left_xyz(3), left_quat(4), left_gripper(1), 
                   right_xyz(3), right_quat(4), right_gripper(1)]
    """
    with h5py.File(h5_file, "r") as fid:
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)  # (n, 2, 3)
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)  # (n, 2, 4) - xyzw
        all_gripper = np.array(fid["state/effector/position"], dtype=np.float32)  # (n, 2)
    
    if slices is None:
        slices = list(range(all_ends_p.shape[0]))
    
    n = len(slices)
    actions = np.zeros([n, 16], dtype=np.float32)
    
    for i, idx in enumerate(slices):
        # 左臂: xyz (3) + quat (4) + gripper (1)
        actions[i, 0:3] = all_ends_p[idx, 0]  # left xyz
        actions[i, 3:7] = all_ends_o[idx, 0]  # left quat (xyzw)
        actions[i, 7] = all_gripper[idx, 0]   # left gripper
        
        # 右臂: xyz (3) + quat (4) + gripper (1)
        actions[i, 8:11] = all_ends_p[idx, 1]  # right xyz
        actions[i, 11:15] = all_ends_o[idx, 1]  # right quat (xyzw)
        actions[i, 15] = all_gripper[idx, 1]    # right gripper
    
    return actions


# ============== Action 条件生成（综合接口） ==============

def generate_action_condition(
    actions: Union[np.ndarray, torch.Tensor],
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    sample_size: Tuple[int, int],
    radius_gen_func=None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    生成 action 条件张量（轨迹图 + 射线图）
    
    这是用于训练和推理的主要接口。
    
    Args:
        actions: action 数据, shape (t, c)
        extrinsics: 相机外参矩阵, shape (v, t, 4, 4)
        intrinsics: 相机内参矩阵, shape (v, 3, 3)
        sample_size: 输出图像尺寸 (h, w)
        radius_gen_func: 可选的半径生成函数
        device: 设备
    
    Returns:
        action 条件张量, shape (9, v, t, h, w)
        其中 9 = 3 (轨迹图 RGB) + 6 (射线图: 3个原点坐标 + 3个方向向量)
    """
    if isinstance(actions, np.ndarray):
        actions = torch.FloatTensor(actions)
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.FloatTensor(extrinsics)
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.FloatTensor(intrinsics)
    
    actions = actions.to(device)
    extrinsics = extrinsics.to(device)
    intrinsics = intrinsics.to(device)
    
    # 计算 w2c (world-to-camera) 矩阵
    w2c = torch.linalg.inv(extrinsics)
    c2w = extrinsics
    
    # 生成轨迹图 (c, v, t, h, w)，c=3
    trajs = get_traj_maps(
        actions, 
        w2c, 
        c2w, 
        intrinsics, 
        sample_size, 
        radius_gen_func=radius_gen_func or simple_radius_gen_func
    )
    
    # 归一化到 [-1, 1]
    trajs = trajs * 2 - 1
    
    # 生成射线图
    v, t = extrinsics.shape[:2]
    h, w = sample_size
    
    rays_o, rays_d = get_ray_maps(
        intrinsics.unsqueeze(dim=1).repeat(1, t, 1, 1).reshape(-1, 3, 3),
        extrinsics.reshape(-1, 4, 4),
        h, w
    )
    
    # (vt, h, w, 6) -> (v, t, h, w, 6) -> (6, v, t, h, w)
    rays = torch.cat((rays_o, rays_d), dim=-1)
    rays = rays.reshape(v, t, h, w, -1)
    rays = rays.permute(4, 0, 1, 2, 3)  # (6, v, t, h, w)
    
    # 拼接轨迹图和射线图
    # trajs: (3, v, t, h, w)
    # rays: (6, v, t, h, w)
    # 输出: (9, v, t, h, w)
    cond_to_concat = torch.cat((trajs.to(device), rays), dim=0)
    return cond_to_concat


def adjust_intrinsic_for_resize_and_crop(
    intrinsic: torch.Tensor,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> torch.Tensor:
    """
    根据 resize + center_crop 变换调整相机内参
    
    视频处理流程（与 ImageCropAndResize 一致）：
    1. 先 resize：scale = max(target_w/orig_w, target_h/orig_h)
    2. 再 center_crop：裁剪到目标尺寸
    
    内参调整：
    1. fx, fy 按 scale 缩放
    2. cx, cy 先按 scale 缩放，再减去 crop 偏移
    
    注意：此函数与 ImageCropAndResize 的实现完全对应：
    - ImageCropAndResize 用 scale = max(...) 确保不变形
    - 内参调整必须与之匹配，否则 3D→2D 投影会错位
    
    Args:
        intrinsic: 原始内参矩阵, shape (3, 3)
        original_size: 原始图像尺寸 (h, w)
        target_size: 目标图像尺寸 (h, w)
    
    Returns:
        调整后的内参矩阵, shape (3, 3)
    """
    orig_h, orig_w = original_size
    tgt_h, tgt_w = target_size
    
    # 计算 resize 的 scale（与 ImageCropAndResize 一致）
    scale = max(tgt_w / orig_w, tgt_h / orig_h)
    
    # resize 后的尺寸
    resized_w = round(orig_w * scale)
    resized_h = round(orig_h * scale)
    
    # center_crop 的偏移
    crop_left = (resized_w - tgt_w) // 2
    crop_top = (resized_h - tgt_h) // 2
    
    # 调整内参
    adjusted = intrinsic.clone()
    
    # fx, fy 按 scale 缩放
    adjusted[0, 0] = intrinsic[0, 0] * scale  # fx
    adjusted[1, 1] = intrinsic[1, 1] * scale  # fy
    
    # cx, cy 先按 scale 缩放，再减去 crop 偏移
    adjusted[0, 2] = intrinsic[0, 2] * scale - crop_left  # cx (ppx)
    adjusted[1, 2] = intrinsic[1, 2] * scale - crop_top   # cy (ppy)
    
    return adjusted


def load_camera_params_from_json(
    intrinsic_path: str,
    extrinsic_path: str,
    frame_indices: List[int] = None,
    original_size: Tuple[int, int] = None,
    target_size: Tuple[int, int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 JSON 文件加载相机参数
    
    内参 JSON 格式:
    {"intrinsic": {"fx": ..., "fy": ..., "ppx": ..., "ppy": ..., ...}}
    
    外参 JSON 格式:
    [{"extrinsic": {"rotation_matrix": [[...], [...], [...]], "translation_vector": [...]}}, ...]
    
    Args:
        intrinsic_path: 内参 JSON 文件路径
        extrinsic_path: 外参 JSON 文件路径
        frame_indices: 要加载的帧索引
        original_size: 原始图像尺寸 (h, w)
        target_size: 目标图像尺寸 (h, w)
    
    Returns:
        intrinsic: 内参矩阵, shape (3, 3)
        extrinsics: 外参矩阵序列, shape (t, 4, 4)
    """
    import json
    
    # 加载内参 JSON
    with open(intrinsic_path, 'r') as f:
        intrinsic_data = json.load(f)
    
    # 解析内参：从 {"intrinsic": {"fx": ..., "fy": ..., "ppx": ..., "ppy": ...}} 格式
    intr = intrinsic_data.get("intrinsic", intrinsic_data)
    fx = intr["fx"]
    fy = intr["fy"]
    ppx = intr["ppx"]  # cx
    ppy = intr["ppy"]  # cy
    
    # 构建 3x3 内参矩阵
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = ppx
    intrinsic[1, 2] = ppy
    
    # 加载外参 JSON
    with open(extrinsic_path, 'r') as f:
        extrinsic_data = json.load(f)
    
    # 解析外参：从 [{"extrinsic": {"rotation_matrix": [...], "translation_vector": [...]}}, ...] 格式
    extrinsics_list = []
    for frame_data in extrinsic_data:
        ext = frame_data.get("extrinsic", frame_data)
        rotation_matrix = np.array(ext["rotation_matrix"], dtype=np.float32)  # 3x3
        translation_vector = np.array(ext["translation_vector"], dtype=np.float32)  # 3
        
        # 构建 4x4 外参矩阵 (camera-to-world)
        extrinsic_matrix = np.eye(4, dtype=np.float32)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector
        extrinsics_list.append(extrinsic_matrix)
    
    extrinsics = torch.FloatTensor(np.stack(extrinsics_list, axis=0))  # (t, 4, 4)
    
    # 如果指定了帧索引，则只取对应帧的外参
    if frame_indices is not None:
        extrinsics = extrinsics[frame_indices]
    
    # 根据 resize + center_crop 调整内参（与 ImageCropAndResize 对应）
    if original_size is not None and target_size is not None:
        orig_h, orig_w = original_size
        tgt_h, tgt_w = target_size
        
        # 计算 resize 的 scale（与 ImageCropAndResize 一致：max 确保不变形）
        scale = max(tgt_w / orig_w, tgt_h / orig_h)
        
        # resize 后的尺寸
        resized_w = round(orig_w * scale)
        resized_h = round(orig_h * scale)
        
        # center_crop 的偏移
        crop_left = (resized_w - tgt_w) // 2
        crop_top = (resized_h - tgt_h) // 2
        
        # fx, fy 按 scale 缩放
        intrinsic[0, 0] = intrinsic[0, 0] * scale  # fx
        intrinsic[1, 1] = intrinsic[1, 1] * scale  # fy
        
        # cx, cy 先按 scale 缩放，再减去 crop 偏移
        intrinsic[0, 2] = intrinsic[0, 2] * scale - crop_left  # cx (ppx)
        intrinsic[1, 2] = intrinsic[1, 2] * scale - crop_top   # cy (ppy)
    
    return intrinsic, extrinsics


def load_camera_params_from_npy(
    intrinsic_path: str,
    extrinsic_path: str,
    frame_indices: List[int] = None,
    original_size: Tuple[int, int] = None,
    target_size: Tuple[int, int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 NPY 文件加载相机参数
    
    与 load_camera_params_from_json 功能相同，但明确从 .npy 文件加载。
    
    Args:
        intrinsic_path: 内参 .npy 文件路径
        extrinsic_path: 外参 .npy 文件路径
        frame_indices: 要加载的帧索引
        original_size: 原始图像尺寸 (h, w)
        target_size: 目标图像尺寸 (h, w)
    
    Returns:
        intrinsic: 内参矩阵, shape (3, 3)
        extrinsics: 外参矩阵序列, shape (t, 4, 4)
    """
    intrinsic = torch.FloatTensor(np.load(intrinsic_path))
    extrinsics = torch.FloatTensor(np.load(extrinsic_path))
    
    # 如果指定了帧索引，则只取对应帧的外参
    if frame_indices is not None and len(extrinsics.shape) > 2:
        extrinsics = extrinsics[frame_indices]
    
    # 根据 resize + center_crop 调整内参（与 ImageCropAndResize 对应）
    if original_size is not None and target_size is not None:
        orig_h, orig_w = original_size
        tgt_h, tgt_w = target_size
        
        # 计算 resize 的 scale（与 ImageCropAndResize 一致：max 确保不变形）
        scale = max(tgt_w / orig_w, tgt_h / orig_h)
        
        # resize 后的尺寸
        resized_w = round(orig_w * scale)
        resized_h = round(orig_h * scale)
        
        # center_crop 的偏移
        crop_left = (resized_w - tgt_w) // 2
        crop_top = (resized_h - tgt_h) // 2
        
        # fx, fy 按 scale 缩放
        intrinsic[0, 0] = intrinsic[0, 0] * scale  # fx
        intrinsic[1, 1] = intrinsic[1, 1] * scale  # fy
        
        # cx, cy 先按 scale 缩放，再减去 crop 偏移
        intrinsic[0, 2] = intrinsic[0, 2] * scale - crop_left  # cx (ppx)
        intrinsic[1, 2] = intrinsic[1, 2] * scale - crop_top   # cy (ppy)
    
    return intrinsic, extrinsics