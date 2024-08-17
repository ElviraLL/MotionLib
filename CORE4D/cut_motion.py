import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from poselib.poselib.skeleton.skeleton3d import (
    SkeletonState,
    SkeletonMotion,
    SkeletonTree,
)


def cut_motion_file(file_path, save_folder, start_percentage=0.0, end_percentage=1.0):
    """
    Cut the motion file to the specified percentage
    Args:
        file_path: str, the path to the motion file
        save_folder: str, the folder to save the cut motion file
        start_percentage: float, the start percentage of the motion file to keep
        end_percentage: float, the end percentage of the motion file to keep
    """
    assert 0 <= start_percentage < end_percentage <= 1, "Invalid percentage range"
    motion_data = np.load(file_path, allow_pickle=True).item()
    # get total number of frames
    N = motion_data["rotation"]["arr"].shape[0]
    print(f"Total number of frames: {N}")
    start_frame = int(start_percentage * N) + 1
    end_frame = int(end_percentage * N)

    return cut_motion_file_by_frame(
        motion_data, file_path, save_folder, start_frame, end_frame
    )


def cut_motion_file_by_frame(
    motion_data, file_path, save_folder, start_frame, end_frame
):
    """
    Cut the motion file to the specified frame range
    Args:
        file_path: str, the path to the motion file
        save_folder: str, the folder to save the cut motion file
        start_frame: int, the start frame of the motion file to keep
        end_frame: int, the end frame of the motion file to keep
    """
    # cut motion data
    for key, value in motion_data.items():
        if key not in ["skeleton_tree", "is_local", "fps", "__name__"]:
            motion_data[key] = value["arr"][start_frame:end_frame]

    skeleton_tree = SkeletonTree.from_dict(motion_data["skeleton_tree"])
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(motion_data["rotation"]),
        torch.from_numpy(motion_data["root_translation"]),
        is_local=motion_data["is_local"],
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, motion_data["fps"])

    sequence_name = file_path.split("/")[-2]
    file_name = file_path.split("/")[-1]
    save_path = os.path.join(save_folder, sequence_name, file_name)
    new_motion.to_file(save_path)
    print(f"new motion data saved to {save_path}")
    return save_path


if __name__ == "__main__":
    cut_motion_file(
        "/home/jing/Documents/projs/CORE4D/isaac_npys/20231002_004/person1_motion.npy",
        "/home/jing/Documents/projs/CORE4D/cutted_isaac_npys",
        0.45,
        0.62,
    )
