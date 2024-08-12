"""
This file is used to read one clip of AMASS data from .npz file.
Visualize the motion data in isaac gym environment.
"""
import numpy as np
from scipy.spatial.transform import Rotation as sRot



fps = 30
data_path = "/home/jing/Documents/AMASS_RAW/ACCAAD/Female1Walking_c3d/B3 - walk1_poses.npz"
entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True)) # load data
framerate = entry_data["mocap_framerate"] # get the framerate
skip = int(framerate / fps)

root_trans = entry_data['trans'][::skip, :] # extract root translation
pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1) # extract pose data axis-angle format.
betas = entry_data['betas']
gender = entry_data['gender']
N = pose_aa.shape[0]

smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES] # Create a mapping of SMPL bones to Mujoco bones.
pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco] # Reshape and reorder the pose data to match the Mujoco order. (this is the order we used in ISAAC GYM)
pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4) # Convert the poses from axis-angle format to quaternion format.
