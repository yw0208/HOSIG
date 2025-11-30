

import os
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
sys.path.append('.')
from utils.joints_to_smplx import JointsToSMPLX
from utils.misc_smplx import smplx_male_model_45
import torch
import pytorch3d.transforms as transforms
from smplx import SMPLXLayer



def global2local_pose(global_pose, parents):
    """
    将全局关节旋转变换为相对于父关节的局部旋转。

    参数:
    global_pose (torch.Tensor): 全局关节旋转矩阵，形状为 (T, J, 3, 3)
    parents (list): 关节的父关节索引列表

    返回:
    torch.Tensor: 相对于父关节的局部旋转矩阵，形状为 (T, J, 3, 3)
    """
    bs, num_joints, _, _ = global_pose.shape
    local_pose = global_pose.clone()

    for jId in range(num_joints):
        parent_id = parents[jId]
        if parent_id >= 0:
            local_pose[:, jId] = torch.matmul(torch.inverse(global_pose[:, parent_id]), global_pose[:, jId])

    return local_pose

def local2global_pose(local_pose, parents):
    # local_pose: T X J X 3 X 3 
    kintree = parents

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 


def get_obj_transformation_in_rwrist(is_grasp, body_model, parents, grasp_fullpose_rotmat_path, grasp_transl_path):
    
    dir_name = os.path.dirname(grasp_fullpose_rotmat_path[0])
    # 计算object跟随手的运动路径
    # 1. 确定object在腕关节坐标下的姿态
    # 加载object位姿
    if is_grasp:
        object_pose = np.load(dir_name+"/batch.npy", allow_pickle=True).item()
        obj_rotation_aa = object_pose['global_orient_obj'][0]
        obj_rotation_rotmat = np.array(transforms.axis_angle_to_matrix(obj_rotation_aa))[0] # 3 X 3 矩阵
        obj_transl = object_pose['transl_obj'][0] # 3维向量
    else:
        object_pose = np.load(dir_name+"/batch.npy", allow_pickle=True).item()
        obj_rotation_aa = object_pose['global_orient_obj'][-1]
        obj_rotation_rotmat = np.array(transforms.axis_angle_to_matrix(obj_rotation_aa))[-1] # 3 X 3 矩阵
        obj_transl = object_pose['transl_obj'][-1] # 3维向量

    # 计算腕关节位置，以及全局旋转
    # 根据grasp pose得到wrist joint位置和hand pose，记得全部转为全局表示，此时得到的是z-up结果
    if is_grasp:
        grasp_fullpose_rotmat = torch.from_numpy(np.load(grasp_fullpose_rotmat_path[0])).cuda()
        grasp_transl = torch.from_numpy(np.load(grasp_transl_path[0])).cuda()
    else:
        grasp_fullpose_rotmat = torch.from_numpy(np.load(grasp_fullpose_rotmat_path[1])).cuda()
        grasp_transl = torch.from_numpy(np.load(grasp_transl_path[1])).cuda()
    
    grasp_smplx_output = body_model(
        global_orient = grasp_fullpose_rotmat[:, 0],
        body_pose = grasp_fullpose_rotmat[:, 1:22],
        left_hand_pose = grasp_fullpose_rotmat[:, -30:-15],                                                                                 
        right_hand_pose = grasp_fullpose_rotmat[:, -15:],
        transl = grasp_transl
    )
    grasp_rw_joint = grasp_smplx_output.joints[0, 21].cpu().numpy() # 这里是obj下的坐标


    # 获得右手的全局旋转表达
    global_grasp_fullpose_rotmat = local2global_pose(grasp_fullpose_rotmat, parents)
    rwrist_global_pose_rotmat = global_grasp_fullpose_rotmat[:, 21:22, :]
    rwh_global_pose_rotmat = rwrist_global_pose_rotmat.cpu().numpy()[0, 0]

    # 计算object在腕关节下的rotation和translation
    rwh_global_pose_rotmat_inv = np.linalg.inv(rwh_global_pose_rotmat)
    obj_rotation_rotmat_in_rwh = np.matmul(rwh_global_pose_rotmat_inv, obj_rotation_rotmat)
    obj_transl_in_rwh = np.matmul(rwh_global_pose_rotmat_inv, (obj_transl - grasp_rw_joint)[None, :].T).T
    
    return obj_rotation_rotmat_in_rwh, obj_transl_in_rwh


def interpolate_rotations_batch(R_start, R_end, num_interpolations=12):
    R_results = np.zeros((R_start.shape[0], num_interpolations, 3, 3))
    for i in range(R_start.shape[0]):
        R_results[i] = interpolate_rotations(R_start[i], R_end[i], num_interpolations)
    
    R_results = R_results.transpose(1, 0, 2, 3)
    return R_results

# ... 保留原有单个插值函数 ...
def interpolate_rotations(R_start, R_end, num_interpolations=12):
    """
    在两个旋转矩阵间进行球面线性插值
    :param R_start: 起始旋转矩阵 (3x3 numpy array)
    :param R_end: 结束旋转矩阵 (3x3 numpy array)
    :param num_interpolations: 总插值数量（包含起始和结束点）
    :return: 插值后的旋转矩阵列表 (num_interpolations, 3, 3)
    """
    # 创建Rotation对象
    key_rots = R.from_matrix([R_start, R_end])
    
    # 创建Slerp插值器
    slerp = Slerp([0, 1], key_rots)
    
    # 生成插值时间点（包含首尾）
    times = np.linspace(0, 1, num_interpolations)
    
    # 执行插值并转换为旋转矩阵
    interp_rots = slerp(times)
    return interp_rots.as_matrix()


def joints2smplx_obj_motion(all_resluts_dict_list, grasp_index, put_index, human_init_transl, obj_start_end_transl, sparse_path, grasp_fullpose_rotmat_path, grasp_transl_path):
    rotation_matrix_yup2zup = np.array([
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]
                    ])

    smplx_male_model = smplx_male_model_45.to(device='cuda')
    joints_to_smplx_model = JointsToSMPLX(opt_rate=0.02, opt_steps=200).to("cuda")
    joints_to_smplx_model.load_and_freeze("./utils/joints_to_smplx/060.pt")

    body_model = SMPLXLayer(
                model_path="body_models/smplx/models",
                gender='male',
                num_pca_comps=45,
                flat_hand_mean=True,
            ).to('cuda')

    # 读取关节树状关系
    npz_data = np.load("body_models/smplx/models/SMPLX_MALE.npz")
    ori_kintree_table = npz_data['kintree_table'] # 2 X 55 
    parents = ori_kintree_table[0, :] # 22 
    parents[0] = -1 # Assign -1 for the root joint's parent idx.
    parents_22=parents[:22]
    
    # 将整个序列组合起来
    joints_seq = []
    global_rh_pose_seq = []
    for index in range(len(all_resluts_dict_list)):
        motions_data = all_resluts_dict_list[index]
        motion_length = 153
        joints = torch.from_numpy(motions_data['motion'][0].transpose(2, 0, 1))[:motion_length]
        global_rh_pose = torch.from_numpy(motions_data['hand_motion'][0, 0, :motion_length])

        if index == 0:
            joints_seq.append(joints)
            global_rh_pose_seq.append(global_rh_pose)
        else:
            # # 对齐当前第2帧与上一段的倒数第1帧，应该先对齐全局位置，再减掉人初始位置
            sparse_path_last_joint = sparse_path[index*4].copy()
            # 对齐
            last_joint_seq_1frames = joints_seq[-1][-1, 0].clone() # 3维向量
            last_joint_seq_1frames[1] = 0. # 保持高度
            joints = joints + last_joint_seq_1frames[None, None, :]
            # -np.array([[[human_init_transl[0][0] ,  0., -human_init_transl[0][1]]]])
            joints_seq.append(joints[1:].float())
            global_rh_pose_seq.append(global_rh_pose[1:])
    
    joints = torch.cat(joints_seq, dim=0)
    global_rh_pose = torch.cat(global_rh_pose_seq, dim=0)

    # 从y坐标转z坐标
    joints=joints[...,[0, 2, 1]]
    joints[..., 1] = -joints[..., 1]
    root_joint = joints[..., 0, :].clone()
    root_joint[..., 2] = 0. # 保持高度
    joints = joints - root_joint[..., None, :]
    joints=joints.reshape(-1, 22*3)

    # 平滑根关节运动
    for i in range(1, len(all_resluts_dict_list)):
        start_joint=root_joint[i*152].cpu().numpy()
        end_joint=root_joint[i*152+11].cpu().numpy()
        # 插值
        inter_joints = np.linspace(start_joint, end_joint, 12)
        root_joint[i*152:i*152+12] = torch.from_numpy(inter_joints)


    # joints to smplx
    params_tensor_list = []
    for joints_start_index in range(0, joints.shape[0], 400):
        joints_clip = joints[joints_start_index:joints_start_index+400]
        joints_mask=torch.zeros(joints_clip.shape[0]).bool().to('cuda').unsqueeze(0)
        params=joints_to_smplx_model.joints_to_params_batch(smplx_male_model,  joints_clip.unsqueeze(0).to('cuda'), joints_mask, optimize=True)
        params_tensor = params[0].to(device='cuda').unsqueeze(0) # 此时得到的这个结果是z-up坐标系下的
        params_tensor_list.append(params_tensor)
    
    params_tensor = torch.cat(params_tensor_list, dim=1)
    params_tensor[0, :, :3] += root_joint.to(params_tensor.device)

    # 将global_rh_pose转换成local_rh_pose
    local_body_pose_aa=params_tensor[..., 3:69].reshape(-1, 22, 3)
    local_body_pose_rotmat=transforms.axis_angle_to_matrix(local_body_pose_aa).reshape(-1, 22, 3, 3)

    # 平滑关节运动
    for i in range(1, len(all_resluts_dict_list)):
        start_rotmat=local_body_pose_rotmat[i*152, :22]
        end_rotmat=local_body_pose_rotmat[i*152+11, :22]
        interp_rotmat=interpolate_rotations_batch(start_rotmat.cpu().numpy(), end_rotmat.cpu().numpy(), 12)
        local_body_pose_rotmat[i*152:i*152+12, :22]=torch.from_numpy(interp_rotmat)

    global_body_pose_rotmat=local2global_pose(local_body_pose_rotmat, parents_22)

    # 将右手全局旋转从y-up坐标系转为z-up坐标系
    global_rh_pose_6d=global_rh_pose.reshape(-1, 6)
    global_rh_pose_rotmat_yup=transforms.rotation_6d_to_matrix(global_rh_pose_6d).reshape(-1, 16, 3, 3)
    rotation_matrix_yup2zup=torch.from_numpy(rotation_matrix_yup2zup).float().unsqueeze(0).unsqueeze(0).repeat(global_rh_pose_rotmat_yup.shape[0], 16, 1, 1)
    global_rh_pose_rotmat = torch.einsum('btij, btjl->btil', rotation_matrix_yup2zup, global_rh_pose_rotmat_yup)
    # 分为腕关节和手关节两部分
    global_rw_pose_rotmat=global_rh_pose_rotmat[..., :1, :, :]
    global_rh_pose_rotmat=global_rh_pose_rotmat[..., 1:, :, :]

    global_body_pose_rotmat[..., 21:22, :, :] = global_rw_pose_rotmat # 腕关节
    global_face_lh_pose_rotmat=torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(global_rh_pose_rotmat.shape[0], 18, 1, 1)

    global_smplx_pose_rotmat=torch.cat([global_body_pose_rotmat.cpu(), global_face_lh_pose_rotmat, global_rh_pose_rotmat], dim=1)
    local_smplx_pose_rotmat=global2local_pose(global_smplx_pose_rotmat, parents)

    # 平滑抓取帧到放下帧的右手腕关节旋转
    start_rotmat=local_smplx_pose_rotmat[grasp_index, 21:22]
    end_rotmat=local_smplx_pose_rotmat[put_index, 21:22]
    interp_rotmat=interpolate_rotations_batch(start_rotmat.cpu().numpy(), end_rotmat.cpu().numpy(), put_index-grasp_index+1)
    local_smplx_pose_rotmat[grasp_index:put_index+1, 21:22]=torch.from_numpy(interp_rotmat)

    # 更新global_rw_pose_rotmat，防止瓶子旋转出错
    global_smplx_pose_rotmat=local2global_pose(local_smplx_pose_rotmat[:, :22], parents_22)
    global_rw_pose_rotmat = global_smplx_pose_rotmat[:, 21:22, :, :]

    local_smplx_pose_aa=transforms.matrix_to_axis_angle(local_smplx_pose_rotmat)
    local_rh_pose_rotmat=local_smplx_pose_rotmat[..., -15:, :, :]
    # 转换成bs*45的轴角形式
    rh_pose_aa=transforms.matrix_to_axis_angle(local_rh_pose_rotmat).reshape(-1, 45)

    body_param = {
        'transl': params_tensor[..., :3].reshape(-1, 3).cpu().numpy(),
        'global_orient': local_smplx_pose_aa[..., :1, :].reshape(-1, 3).cpu().numpy(),
        'body_pose': local_smplx_pose_aa[..., 1:22, :].reshape(-1, 63).cpu().numpy(),
        'right_hand_pose': rh_pose_aa.reshape(-1, 45).cpu().numpy(),
        'global_rhw_pose_rotmat':transforms.matrix_to_rotation_6d(global_rh_pose_rotmat_yup).reshape(-1, 96).cpu().numpy(),
        'local_smplx_pose_rotmat': local_smplx_pose_rotmat.cpu().numpy(),
    }
    
    # 把抓取帧到放下帧的右手固定下来
    body_param['right_hand_pose'][grasp_index+1:put_index+1] = body_param['right_hand_pose'][grasp_index:grasp_index+1]

    # 重新计算手腕的关节位置
    new_smplx_output = body_model(
        global_orient = local_body_pose_rotmat[:, 0].cuda(),
        body_pose = local_body_pose_rotmat[:, 1:22].cuda(),
        right_hand_pose = local_rh_pose_rotmat.cuda(),
        transl = params_tensor[..., :3].reshape(-1, 3).cuda()
    )
    rw_joint = new_smplx_output.joints[:, 21].cpu().numpy() # 这里是z-up结果,在初始人的坐标系下
    body_param['joints']=new_smplx_output.joints[:, :22].cpu().numpy()

    # # 以抓取帧为标准计算object的整个运动序列
    obj_rotation_rotmat_in_rwh_grasp, obj_transl_in_rwh_grasp = get_obj_transformation_in_rwrist(True, body_model, parents, grasp_fullpose_rotmat_path, grasp_transl_path)
    # obj_global_orient_seq_grasp = np.matmul(global_rw_pose_rotmat, obj_rotation_rotmat_in_rwh).cpu().numpy()
    # obj_transl_seq_grasp = (np.matmul(global_rw_pose_rotmat[:, 0], obj_transl_in_rwh.T)[:,:, 0] + rw_joint).cpu().numpy()

    # # 以放下帧为标准计算object的整个运动序列
    obj_rotation_rotmat_in_rwh_put, obj_transl_in_rwh_put = get_obj_transformation_in_rwrist(False, body_model, parents, grasp_fullpose_rotmat_path, grasp_transl_path)
    # obj_global_orient_seq_put = np.matmul(global_rw_pose_rotmat, obj_rotation_rotmat_in_rwh).cpu().numpy()
    # obj_transl_seq_put = (np.matmul(global_rw_pose_rotmat[:, 0], obj_transl_in_rwh.T)[:,:, 0] + rw_joint).cpu().numpy()
    
    # # 利用插值计算从抓取帧到放下帧的object的整个运动序列
    # 先初始化
    obj_global_orient_seq = np.zeros((len(body_param['transl']), 3, 3))
    obj_transl_seq = np.zeros((len(body_param['transl']), 3))

    # 对object旋转进行插值
    interp_obj_rotmat=interpolate_rotations_batch(obj_rotation_rotmat_in_rwh_grasp[None, ...], obj_rotation_rotmat_in_rwh_put[None, ...], put_index-grasp_index+1)
    inter_obj_translation = np.linspace(obj_transl_in_rwh_grasp[0],  obj_transl_in_rwh_put[0], put_index-grasp_index+1)
    # 从右手坐标系转换到世界坐标系
    obj_global_orient_seq[grasp_index:put_index+1] = np.einsum('ijk,ikl->ijl', global_rw_pose_rotmat[grasp_index:put_index+1, 0], interp_obj_rotmat[:, 0])
    obj_transl_seq[grasp_index:put_index+1] = np.einsum('ijk,ikl->ijl', global_rw_pose_rotmat[grasp_index:put_index+1, 0], inter_obj_translation[..., None])[:,:, 0] + rw_joint[grasp_index:put_index+1]

    # 初始化
    object_pose = np.load(os.path.dirname(grasp_fullpose_rotmat_path[0])+"/batch.npy", allow_pickle=True).item()

    # 将抓取帧之前的object位姿都设置为初始状态
    obj_rotation_aa = object_pose['global_orient_obj'][0]
    obj_rotation_rotmat = np.array(transforms.axis_angle_to_matrix(obj_rotation_aa))[0] # 3 X 3 矩阵
    obj_transl = object_pose['transl_obj'][0] # 3维向量
    obj_global_orient_seq[:grasp_index] = obj_rotation_rotmat[None, :]
    obj_transl_seq[:grasp_index] = obj_transl[None, :] + np.array(obj_start_end_transl[0])-np.array(human_init_transl)# 加上原本的obj transl对齐全局空间，再减掉human transl对齐人的空间

    # 将放下帧之后的object位姿都设置为结束状态
    obj_transl = object_pose['transl_obj'][-1] # 3维向量读取最后的obj位置
    obj_rotation_aa = object_pose['global_orient_obj'][-1]
    obj_rotation_rotmat = np.array(transforms.axis_angle_to_matrix(obj_rotation_aa))[-1] # 3 X 3 矩阵
    obj_global_orient_seq[put_index+1:] = obj_rotation_rotmat[None, :]
    obj_transl_seq[put_index+1:] = obj_transl[None, :] + np.array(obj_start_end_transl[1])-np.array(human_init_transl)# 加上原本的obj transl对齐全局空间，再减掉human transl对齐人的空间

    # # 将抓取帧到放下帧的object位姿前一半用抓取动作的object位姿，后一半用放下动作的object位姿
    # obj_global_orient_seq[put_index+1-(put_index-grasp_index)//2 : put_index+1] = obj_global_orient_seq_put[put_index+1-(put_index-grasp_index)//2 : put_index+1]
    # obj_transl_seq[put_index+1-(put_index-grasp_index)//2 : put_index+1] = obj_transl_seq_put[put_index+1-(put_index-grasp_index)//2 : put_index+1]
    
    body_param['obj_global_orient'] = obj_global_orient_seq.reshape(-1, 3, 3)
    body_param['obj_transl'] = obj_transl_seq.reshape(-1, 3)
    np.savez(os.path.dirname(grasp_fullpose_rotmat_path[0])+'/sample_w_scene_guide.npz', **body_param)
    print("save to ", os.path.dirname(grasp_fullpose_rotmat_path[0])+'/sample_w_scene_guide.npz') 

if __name__=='__main__':
    joints2smplx_obj_motion()

