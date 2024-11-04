# This script reads Hi4D files and converts them to .npz files that AMP's MotionLib can read.
# Step 1: Read core4d files
# Step 2: Process data
# Step 3: Write data to .npz files

import os
from pathlib import Path
import sys
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp

import torch
from typing import Union
from tqdm import tqdm
import pickle

# Add the parent directory to the system path to import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import necessary modules from other files
from CORE4D.read_core4d_data import read_core4D_data
from poselib.poselib.skeleton.skeleton3d import (
    SkeletonState,
    SkeletonMotion,
    SkeletonTree,
)
from conversion.smpl_to_isaac import *
from conversion.const import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_ISAAC_NAMES_24,
    SMPLX_BONE_ORDER_NAMES,
    SMPL2SMPLX,
)

def read_hi4d_data(file_path, idx):
    """Read the Hi4D test or train pickle file"""
    with open(file_path, "rb") as f:
        data_total = pickle.load(f)
        data = data_total[idx]
        num_frames = len(data)
        
        # Initialize lists to store motion data for two persons
        poses_1 = []
        trans_1 = []
        smpl_joints_3d_1 = []
        poses_2 = []
        trans_2 = []
        smpl_joints_3d_2 = []

        # Extract motion data for each frame
        for f in range(num_frames):
            img_path = data[f]["img_path"]
            h_w = data[f]["h_w"]
            person_0 = data[f]["0"]
            poses_1.append(person_0["pose"])
            trans_1.append(person_0["trans"])
            smpl_joints_3d_1.append(person_0["smpl_joints_3d"])
            person_1 = data[f]["1"]
            poses_2.append(person_1["pose"])
            trans_2.append(person_1["trans"])
            smpl_joints_3d_2.append(person_1["smpl_joints_3d"])

    # Convert lists to numpy arrays and return as dictionaries
    person_1_motion = {
        "poses": np.array(poses_1),
        "trans": np.array(trans_1),
        "smpl_joints_3d": np.array(smpl_joints_3d_1),
    }
    person_2_motion = {
        "poses": np.array(poses_2),
        "trans": np.array(trans_2),
        "smpl_joints_3d": np.array(smpl_joints_3d_2),
    }
    return person_1_motion, person_2_motion


def interpolate_motion(root_translations, local_rotations, current_fps, target_fps):
    """
    Interpolates the motion data from current_fps to target_fps.

    Args:
        root_translations (np.array): Shape (N, J, 3), root translations for N frames, J joints.
        local_rotations (np.array): Shape (N, J, 4), local rotations in quaternion for N frames, J joints.
        current_fps (int): Current frames per second of the motion data.
        target_fps (int): Target frames per second for the interpolation.

    Returns:
        tuple: interpolated_root_translations, interpolated_local_rotations
    """
    # Calculate the number of current frames and the desired number of frames
    num_current_frames = root_translations.shape[0]
    num_target_frames = int(num_current_frames * (target_fps / current_fps))
    duration = num_current_frames / current_fps
    original_timestamps = np.linspace(0, num_current_frames / current_fps, num_current_frames, endpoint=False)
    target_timestamps = np.linspace(0, duration, num_target_frames, endpoint=False)


    # Create new frame indices for interpolation
    current_frame_indices = np.linspace(0, num_current_frames - 1, num=num_current_frames)
    target_frame_indices = np.linspace(0, num_current_frames - 1, num=num_target_frames)

    # Interpolate root translations using linear interpolation        
    interpolation_func = interp1d(original_timestamps, root_translations, axis=0, kind='linear', fill_value="extrapolate")
    interpolated_root_translations = interpolation_func(target_timestamps)

    # Prepare rotation interpolation
    # Process each joint separately
    num_joints = local_rotations.shape[1]
    interpolated_local_rotations = np.zeros((num_target_frames, num_joints, 4))
    for joint in range(num_joints):
        # Extract quaternions for the current joint across all frames
        joint_quaternions = local_rotations[:, joint, :]
        
        # Create a Rotation object from the joint quaternions
        rotation_obj = sRot.from_quat(joint_quaternions)
        
        # Initialize the Slerp interpolator with original times and rotations
        slerp_interpolator = Slerp(current_frame_indices, rotation_obj)
        
        # Interpolate to find the rotations at the target times
        interpolated_rotations = slerp_interpolator(target_frame_indices)
        
        # Convert the interpolated rotations back to quaternions
        interpolated_local_rotations[:, joint, :] = interpolated_rotations.as_quat()

    return interpolated_root_translations, interpolated_local_rotations


def convert_yup_to_zup_translation(
    root_translations: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert the root translations from Y-up to Z-up coordinate system.
    Args:
        root_translations: [N, 3] numpy array
    """
    # Swap Y and Z axes and invert the new Y axis
    root_translations[:, [1, 2]] = root_translations[:, [2, 1]]
    root_translations[:, 1] *= -1
    return root_translations


def process_single_person(motion_data, save_path, upright_start=True):
    smpl_joints_3d = motion_data["smpl_joints_3d"]  # [N_frames, 24, 3] 140, 24, 3
    poses = motion_data["poses"]  # [N_frames, 72] # 140, 72
    root_trans = motion_data["trans"]  # [N_frames, 3] # 140, 3
    global_orient = poses[:, :3]
    num_joints = smpl_joints_3d.shape[1]

    N_frames = smpl_joints_3d.shape[0]
    pose_aa = poses.reshape(N_frames, -1, 3)
    N = pose_aa.shape[0]
    # smplx_body_joint_names = SMPLX_BONE_ORDER_NAMES

    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']


    smplx_2_isaac = [SMPL_BONE_ORDER_NAMES.index(q) for q in mujoco_joint_names if q in SMPL_BONE_ORDER_NAMES] 
    pose_aa_isaac = pose_aa[:, smplx_2_isaac]
    pose_quat_isaac = (
        sRot.from_rotvec(pose_aa_isaac.reshape(-1, 3)).as_quat().reshape(N, num_joints, 4)
    )  # TODO: why reshape here?
    
    # interpolate the motion data to 30 fps
    # root_trans, pose_quat_isaac = interpolate_motion(root_trans, pose_quat_isaac, 15, 30)
    # N = root_trans.shape[0]

    skeleton_tree = SkeletonTree.from_mjcf(
        "./conversion/data/mjcf/smpl_humanoid_24.xml"
    )  
    root_trans_offset = (
        torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
    )  # TODO: why add offset here? #[234, 3]

    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_isaac),
        root_trans_offset,
        is_local=True,
    )

    root_trans_offset = convert_yup_to_zup_translation(root_trans_offset)

    if upright_start:
        # this is for AMASS, we don't necessary need this for CORE4D, check if we need this
        # from_quat: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # Why rotate all global_rotation matrices around (1, 1, 1) for 120 degree?
        pose_quat_global = (
            (
                sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            )
            .as_quat()
            .reshape(N, -1, 4)
        )

        rot_x_90 = sRot.from_euler(
            "x", 90, degrees=True
        )  # rotate around x axis for 90 degree
        pose_quat_global = (
            (rot_x_90 * sRot.from_quat(pose_quat_global.reshape(-1, 4)))
            .as_quat()
            .reshape(N, -1, 4)
        )

        rot_x_180 = sRot.from_euler("x", 180, degrees=True)
        # rotate the root joint 180 degree around x axis
        pose_quat_global = (
            (rot_x_180 * sRot.from_quat(pose_quat_global.reshape(-1, 4)))
            .as_quat()
            .reshape(N, -1, 4)
        )

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False,
        )

    global_translation = new_sk_state.global_transformation[
        :, :, 4:
    ]  # global transformation: [N, 19, 7]
    min_z = global_translation[:, :, 2].min()  # min for all frames and all joints
    root_trans_offset[:, 2] -= min_z
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        root_trans_offset,
        is_local=False,
    )

    motion = SkeletonMotion.from_skeleton_state(new_sk_state, 30)

    motion.to_file(save_path)
    print(f"new motion data saved to {save_path}")

if __name__ == "__main__":
    # Define file path and read motion data
    file_path = "/home/jing/Downloads/Hi4D/images/Hi4D/annot/test.pkl"
    person_1_motion, person_2_motion = read_hi4d_data(file_path, 0)
    
    # Process and save motion data for person 1
    process_single_person(person_1_motion, "./test_1.npy")
    # Uncomment to process person 2
    process_single_person(person_2_motion, "./test_2.npy")
