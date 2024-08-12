# this scripts reads core4d files and convert them to .npz files that AMP's MotionLib can read
# step1: read core4d files
# step2: process data
# write data to .npz files
import os
import sys
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from CORE4D.read_core4d_data import read_core4D_data
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion, SkeletonTree
from conversion.smpl_to_isaac import *
from conversion.const import SMPL_BONE_ORDER_NAMES, SMPL_ISAAC_NAMES, SMPLX_BONE_ORDER_NAMES

def process_core4d_data(motion_data_folder, sequence_name, save_folder):
    data_folder = os.path.join(motion_data_folder, 'human_object_motions', sequence_name)
    person_1_path = os.path.join(data_folder, 'person1_poses.npz')
    person_2_path = os.path.join(data_folder, 'person2_poses.npz')
    person_1_save_path = os.path.join(save_folder, sequence_name, 'person1_motion.npz')
    person_2_save_path = os.path.join(save_folder, sequence_name, 'person2_motion.npz')
    process_single_person(person_1_path, person_1_save_path)
    process_single_person(person_2_path, person_2_save_path)


def process_single_person(person_path, save_path, upright_start=False):
    entry_data = read_core4D_data(person_path)
    N_frame = entry_data['betas'].shape[0]
    
    # read raw data
    fps = 30 # TODO: figure out the actual fps, assuming it is 30
    skip = 1 # sampling rate
    root_trans = entry_data['transl'][::skip, :] # [N_frame, 3], root's position
    body_pose_aa = entry_data['body_pose'][::skip, :] # [N_frame, 21, 3], body pose in axis-angle # TODO: do I need to add root?
    root_pose_aa = np.zeros((root_trans.shape[0], 6)) # TODO: why 6?
    pose_aa = np.concatenate([body_pose_aa, root_pose_aa], axis=-1)
    N = pose_aa.shape[0]

    # prepare joint rotation in target joint order (19 bones)
    smpl_bone_order_names = SMPLX_BONE_ORDER_NAMES[:22] # no hand (diff from amass, they have hands thus 24 joints)
    smplx_2_mujoco = [smpl_bone_order_names.index(q) for q in SMPL_ISAAC_NAMES if q in smpl_bone_order_names]
    pose_aa_isaac = pose_aa.reshape(N, 24, 3)[:, smplx_2_mujoco]
    pose_quat_isaac = sRot.from_rotvec(pose_aa_isaac.reshape(-1, 3)).as_quat().reshape(N, 24, 4) # TODO: why reshape here? 

    skeleton_tree = SkeletonTree.from_mjcf('./conversion/data/mjcf/smpl_humanoid_19.xml') # 19 bone's skeleton_tree
    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0] # TODO: why add offset here?
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_isaac), root_trans_offset, is_local=True)
    
    
    if upright_start:
        # this is for AMASS, we don't necessary need this for CORE4D
        # from_quat: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # Why rotate all global_rotation matrices around (1, 1, 1) for 120 degree?
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  
        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
    
    
    motion = SkeletonMotion.from_skeleton_state(new_sk_state, 30)
    motion.to_file(save_path)
    
    





