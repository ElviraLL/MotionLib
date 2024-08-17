# this scripts reads core4d files and convert them to .npz files that AMP's MotionLib can read
# step1: read core4d files
# step2: process data
# write data to .npz files
import os
from pathlib import Path
import sys
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
from typing import Union
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from CORE4D.read_core4d_data import read_core4D_data
from poselib.poselib.skeleton.skeleton3d import (
    SkeletonState,
    SkeletonMotion,
    SkeletonTree,
)
from conversion.smpl_to_isaac import *
from conversion.const import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_ISAAC_NAMES,
    SMPLX_BONE_ORDER_NAMES,
    SMPL2SMPLX,
)


def process_core4d_data(motion_data_folder, sequence_name, save_folder):
    data_folder = os.path.join(motion_data_folder, sequence_name)
    person_1_path = os.path.join(data_folder, "person1_poses.npz")
    person_2_path = os.path.join(data_folder, "person2_poses.npz")
    tgt_folder = os.path.join(save_folder, sequence_name.replace("/", "_"))
    os.makedirs(tgt_folder, exist_ok=True)
    person_1_save_path = os.path.join(tgt_folder, "person1_motion.npy")
    person_2_save_path = os.path.join(tgt_folder, "person2_motion.npy")
    process_single_person(person_1_path, person_1_save_path)
    process_single_person(person_2_path, person_2_save_path)


def convert_yup_to_zup_translation(
    root_translations: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert the root translations from yup to zup
    Args:
        root_translations: [N, 3] numpy array
    """
    root_translations[:, [1, 2]] = root_translations[:, [2, 1]]
    root_translations[:, 1] *= -1
    return root_translations


def process_single_person(person_path, save_path, upright_start=True):
    """
    We need upright_start in our case
    """
    entry_data = read_core4D_data(person_path)
    N_frame = entry_data["betas"].shape[0]

    # read raw data
    fps = 30  # TODO: figure out the actual fps, assuming it is 30
    skip = 1  # sampling rate

    root_trans = entry_data["transl"][::skip, :]  # [N_frame, 3], root's position
    root_rot_aa = np.expand_dims(
        entry_data["global_orient"][::skip, :], axis=1
    )  # [N_frame, 1,3], root's orientation in axis-angle
    body_pose_aa = entry_data["body_pose"][
        ::skip, :
    ]  # [N_frame, 21, 3], body pose in axis-angle # TODO: do I need to add root?
    # first concatenate the root pose, then select the useful joints (we don't need head, toe), there is no hand data in SMPLX's motion data
    pose_aa = np.concatenate(
        [root_rot_aa, body_pose_aa], axis=1
    )  # [N_frame, 22, 3] (24 - hands),
    N = pose_aa.shape[0]

    # prepare joint rotation in target joint order (19 bones)
    smplx_body_joint_names = SMPLX_BONE_ORDER_NAMES[
        :22
    ]  # no hand (diff from amass, they have hands thus 24 joints)
    smplx_2_isaac = [
        smplx_body_joint_names.index(SMPL2SMPLX[q])
        for q in SMPL_ISAAC_NAMES
        if SMPL2SMPLX[q] in smplx_body_joint_names
    ]
    pose_aa_isaac = pose_aa[:, smplx_2_isaac]
    pose_quat_isaac = (
        sRot.from_rotvec(pose_aa_isaac.reshape(-1, 3)).as_quat().reshape(N, 19, 4)
    )  # TODO: why reshape here?

    skeleton_tree = SkeletonTree.from_mjcf(
        "./conversion/data/mjcf/smpl_humanoid_19.xml"
    )  # 19 bone's skeleton_tree
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
    # print(f"new motion data saved to {save_path}")
    data = np.load(save_path, allow_pickle=True)
    # print("finished")


def process_all_data(motion_data_folder, save_folder):
    motion_data_folder = Path(motion_data_folder)
    outer_loop_count = len(list(motion_data_folder.iterdir()))
    for time_step in tqdm(motion_data_folder.iterdir(), desc="Outer Loop"):
        if time_step.is_dir():
            inner_progress = tqdm(desc="Inner Loop", leave=False)
            for sequence in time_step.iterdir():
                if sequence.is_dir():
                    sequence_name = time_step.name + "/" + sequence.name
                    process_core4d_data(motion_data_folder, sequence_name, save_folder)
                    inner_progress.update(1)


if __name__ == "__main__":
    motion_data_folder = "/home/jing/Documents/projs/CORE4D/human_object_motions"
    # sequence_name = '20231002/003'
    save_folder = "/home/jing/Documents/projs/CORE4D/isaac_npys"
    # process_core4d_data(motion_data_folder, sequence_name, save_folder)
    process_all_data(motion_data_folder, save_folder)
