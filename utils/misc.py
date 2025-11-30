import torch

def compute_repr_dimesion(data_repr: str) -> int:
    if data_repr == 'smplx_no_hands':
        return 3 + 3 + 63 # global translation + global orientation + 21 joints rotation
    elif data_repr == 'pos':
        return 22 * 3
    elif data_repr == 'pos_rot':
        return 22 * 3 + 21 * 3 # 22 joints position and 21 joints rotation
    elif data_repr == 'contact_one_joints':
        return 1
    elif data_repr == 'contact_all_joints':
        return 22
    elif data_repr == 'contact_cont_joints':
        return 6
    elif data_repr == 'contact_pelvis':
        return 1
    elif data_repr == 'h3d':
        return 263
    else:
        raise ValueError(f"Unknown data representation: {data_repr}")
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen
