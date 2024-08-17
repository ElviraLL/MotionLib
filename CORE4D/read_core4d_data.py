"""
CORE4D dataset is recorded in smplx's format skeleton:
     0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'jaw',
    23: 'left_eye',
    24: 'right_eye',
    25: 'left_index1',
    26: 'left_index2',
    27: 'left_index3',
    28: 'left_middle1',
    29: 'left_middle2',
    30: 'left_middle3',
    31: 'left_pinky1',
    32: 'left_pinky2',
    33: 'left_pinky3',
    34: 'left_ring1',
    35: 'left_ring2',
    36: 'left_ring3',
    37: 'left_thumb1',
    38: 'left_thumb2',
    39: 'left_thumb3',
    40: 'right_index1',
    41: 'right_index2',
    42: 'right_index3',
    43: 'right_middle1',
    44: 'right_middle2',
    45: 'right_middle3',
    46: 'right_pinky1',
    47: 'right_pinky2',
    48: 'right_pinky3',
    49: 'right_ring1',
    50: 'right_ring2',
    51: 'right_ring3',
    52: 'right_thumb1',
    53: 'right_thumb2',
    54: 'right_thumb3'
"""

import numpy as np
import smplx
import torch


def read_core4D_data(file_path):
    """
    Read the core4D data from the .npz file.
    data dictionary keys: vertices, joints, betas, expression, global_orient, transl, body_pose, left_hand_pose, right_hand_pose
        vertices:         [N_frame, 10475, 3], numpy.float32, denotes the human's SMPLX vertex positions in each frame.
        joints:           [N_frame, 127, 3],   numpy.float32, denotes the human's SMPLX joint positions in each frame.
        betas:            [N_frame, 10],       numpy.float32, denotes the human shape parameter in each frame. In fact, the shape parameters are exactly the same among all the frames.
        expression:       [N_frame, 10]        numpy.float32, denotes the human expression parameter in each frame.
        global_orient:    [N_frame, 3]         numpy.float32, denotes the human root's orientation in each frame. The orientation is represented as axis-angle.
        transl:           [N_frame, 3]         numpy.float32, denotes the human root's position in each frame.
        body_pose:        [N_frame, 21, 3]     numpy.float32, denotes the human's body pose in each frame. The local orientation of each joint is represented as axis-angle.
        left_hand_pose:   [N_frame, 12]        numpy.float32, denotes the human's left hand pose in each frame. The hand pose is defined in PCA space with 12 DoF.
        right_hand_pose:  [N_frame, 12]        numpy.float32, denotes the human's right hand pose in each frame. The hand pose is defined in PCA space with 12 DoF.
    """
    data = dict(np.load(open(file_path, "rb"), allow_pickle=True))["arr_0"].item()
    return data


def forward_smplx(human_motion):
    """
    Forward the SMPLX model to get the vertices and joints.
    Args:
        human_motion: dict, the human motion data read from CORE4D dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_frame = human_motion["betas"].shape[0]

    # create SMPLX model
    smplx_model = smplx.create(
        "./data",
        model_type="smplx",
        gender="neutral",
        batch_size=N_frame,
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="pkl",
        use_pca=True,
        num_pca_comps=12,
        flat_hand_mean=True,
    )
    smplx_model.to(device)

    # prepare SMPLX parameters
    SMPLX_params = {
        "betas": torch.from_numpy(human_motion["betas"]).to(device),
        "global_orient": torch.from_numpy(human_motion["global_orient"]).to(device),
        "transl": torch.from_numpy(human_motion["transl"]).to(device),
        "body_pose": torch.from_numpy(human_motion["body_pose"]).to(device),
        "left_hand_pose": torch.from_numpy(human_motion["left_hand_pose"]).to(device),
        "right_hand_pose": torch.from_numpy(human_motion["right_hand_pose"]).to(device),
        "expression": torch.from_numpy(human_motion["expression"]).to(device),
    }

    # SMPLX forward
    results = smplx_model(
        betas=SMPLX_params["betas"],
        body_pose=SMPLX_params["body_pose"],
        global_orient=SMPLX_params["global_orient"],
        transl=SMPLX_params["transl"],
        left_hand_pose=SMPLX_params["left_hand_pose"],
        right_hand_pose=SMPLX_params["right_hand_pose"],
    )
    print(results)


if __name__ == "__main__":
    file_path = "/home/jing/Documents/projs/ASE/data/CORE4D/human_object_motions/20231002/003/person1_poses.npz"
    data = read_core4D_data(file_path)
    forward_smplx(data)
