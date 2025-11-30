import matplotlib.pyplot as plt
import numpy as np
from smplx import SMPLXLayer
import torch
from pytorch3d import transforms
import sys
sys.path.append('.')


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

def plot_3d(joints):
    joints=joints.reshape(-1, 22, 3)

    for i in range(joints.shape[0]):

        points = joints[i]
        fig = plt.figure()

        # 添加一个3D子图
        ax = fig.add_subplot(111, projection='3d')

        # 绘制三维坐标点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        # 设置坐标轴标签
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

         # 设置3D图形的长宽高比例
        ax.axis('equal')

        # 显示图形
        plt.savefig('3d_points.jpg')



# 读取关节树状关系
npz_data = np.load("body_models/smplx/models/SMPLX_MALE.npz")
ori_kintree_table = npz_data['kintree_table'] # 2 X 55 
parents = ori_kintree_table[0, :] # 22 
parents[0] = -1 # Assign -1 for the root joint's parent idx.

body_model = SMPLXLayer(
            model_path="body_models/smplx/models",
            gender='male',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to('cuda')

rotation_matrix_yup2zup = np.array([
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]
                    ])

rotation_matrix_zup2yup = np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]
                    ])

def get_hint_for_generate(trajectory, t, grasp_flag, grasp_time, grasp_index, start_index, obj_transl, grasp_fullpose_rotmat_path, grasp_transl_path, sparse_path):
    
    # 初始化hint
    hint = np.zeros((196, 162))

    # 施加根关节轨迹约束，包括第0, 38, 76, 114, 152帧
    for i in range(5):
        hint[i*38, :3] = trajectory[i] - t # 将起点归0

    # 对grasp帧施加约束
    if grasp_flag:
        # 抓住了，根据是第几次抓住，来load grasp pose
        if grasp_time == 1:
            grasp_fullpose_rotmat = np.load(grasp_fullpose_rotmat_path[0])
            grasp_transl = np.load(grasp_transl_path[0])
        elif grasp_time == 2:
            grasp_fullpose_rotmat = np.load(grasp_fullpose_rotmat_path[1])
            grasp_transl = np.load(grasp_transl_path[1])
        
        # 根据grasp pose得到wrist joint位置和hand pose，记得全部转为全局表示，此时得到的是z-up结果
        grasp_fullpose_rotmat = torch.from_numpy(grasp_fullpose_rotmat).cuda()
        grasp_transl = torch.from_numpy(grasp_transl).cuda()
        grasp_smplx_output = body_model(
            global_orient = grasp_fullpose_rotmat[:, 0],
            body_pose = grasp_fullpose_rotmat[:, 1:22],
            left_hand_pose = grasp_fullpose_rotmat[:, -30:-15],                                                                                 
            right_hand_pose = grasp_fullpose_rotmat[:, -15:],
            transl = grasp_transl + torch.from_numpy(obj_transl[grasp_time-1]).cuda()
        )
        grasp_joint = grasp_smplx_output.joints[0, :22].cpu().numpy() # 这里是z-up下的全局坐标

        # 获得右手的全局旋转表达
        global_grasp_fullpose_rotmat = local2global_pose(grasp_fullpose_rotmat, parents)
        rh_global_pose_rotmat = global_grasp_fullpose_rotmat[:, -15:, :]
        rwrist_global_pose_rotmat = global_grasp_fullpose_rotmat[:, 21:22, :]
        rwh_global_pose_rotmat = torch.cat([rwrist_global_pose_rotmat, rh_global_pose_rotmat], dim=1).cpu()

        # 全部转y-up
        grasp_joint_yup = (rotation_matrix_zup2yup@grasp_joint.T).T
        rotmat_zup2yup=torch.from_numpy(rotation_matrix_zup2yup).float().unsqueeze(0).unsqueeze(0).repeat(rwh_global_pose_rotmat.shape[0], 16, 1, 1)
        rwh_global_pose_rotmat_yup = torch.einsum('btij, btjl->btil', rotmat_zup2yup, rwh_global_pose_rotmat)
        rwh_global_pose_6d_yup = transforms.matrix_to_rotation_6d(rwh_global_pose_rotmat_yup).reshape(96)

        # 对抓住那一帧施加根、腕关节和手指pose约束，额外在周围10帧也加上腕关节约束
        grasp_joint_yup=grasp_joint_yup - t # 对齐起点
        hint[grasp_index*38, :3] = grasp_joint_yup[0]
        hint[grasp_index*38, 63:66] = grasp_joint_yup[-1]
        hint[grasp_index*38, 66:] = rwh_global_pose_6d_yup

        if grasp_index < 4: # 判断是不是最后一段动作放下的
            for i in range(grasp_index*38-5, grasp_index*38+6):
                hint[i, 63:66] = grasp_joint_yup[-1]
                # hint[i, 66:72] = rwh_global_pose_6d_yup[:6]
        else:
            for i in range(grasp_index*38-10, grasp_index*38+1):
                hint[i, 63:66] = grasp_joint_yup[-1]
                # hint[i, 66:72] = rwh_global_pose_6d_yup[:6]


    if start_index == 0:
        pass
        # # 实现从坐着到站起来，对1-10帧施加预设好的根关节+双脚关节约束+头部约束
        # ori_root_joint = np.array([[0.92243, 0.57106+0.1, 2.5346]]) - t
        # ori_lf_joint = np.array([[0.47137, 0.02, 2.6489]]) - t
        # ori_rf_joint = np.array([[0.47137, 0.02, 2.3541]]) - t
        # ori_head_joint = np.array([[0.92243, 0.57106+0.6+0.1, 2.5346]]) - t

        # # 10 * 3
        # ori_root_joint = np.repeat(ori_root_joint, 10, axis=0)
        # ori_lf_joint = np.repeat(ori_lf_joint, 10, axis=0)
        # ori_rf_joint = np.repeat(ori_rf_joint, 10, axis=0)
        # ori_head_joint = np.repeat(ori_head_joint, 10, axis=0)

        # # 施加约束
        # hint[76-5:76+5, :3] = ori_root_joint
        # hint[76-5:76+5, 30:33] = ori_lf_joint
        # hint[76-5:76+5, 33:36] = ori_rf_joint
        # hint[76-5:76+5, 45:48] = ori_head_joint

    else: 
        # 另一种做法，取上一段最后一帧动作的结果施加给当前前11帧
        motions_data=np.load('save/scomogen/samples_scomogen_000875000_seed10_predefined/results.npy', allow_pickle=True)
        motions_data = dict(motions_data.item())
        motion_length = 153
        joints = torch.from_numpy(motions_data['motion'][0].transpose(2, 0, 1))[:motion_length]

        last_joint_seq_1frames = joints[-1:] # 这里是y-up结果,在初始人的坐标系下
        last_joint_seq_1frames_t = np.array([last_joint_seq_1frames[0, 0, 0], 0., last_joint_seq_1frames[0, 0, 2]]) # 起点
        last_joint_seq_1frames_yup = last_joint_seq_1frames - last_joint_seq_1frames_t # 对齐起点
        
        # 对前10帧施加关节约束
        # hint[1:11, :66] = last_joint_seq_1frames_yup.reshape(1, 66).repeat(10, 1)
        hint[1:2, :66] = last_joint_seq_1frames_yup.reshape(1, 66)
        hint[10:11, :66] = last_joint_seq_1frames_yup.reshape(1, 66)


    # 坐下专用
    # if start_index >= len(sparse_path)-8:
    #     # pass
    #     # 实现从坐着到站起来，对1-10帧施加预设好的根关节+双脚关节约束
    #     ori_root_joint = np.array([[-0.3519, 0.7, 2.1457]]) - t
    #     ori_lf_joint = np.array([[-0.14483, 0.02, 2.4347+0.2]]) - t
    #     ori_rf_joint = np.array([[-0.40382, 0.02, 2.4347+0.2]]) - t
    #     ori_head_joint = np.array([[-0.3519, 0.7, 2.1457]]) - t

    #     # 10 * 3
    #     ori_root_joint = np.repeat(ori_root_joint, 1, axis=0)
    #     ori_lf_joint = np.repeat(ori_lf_joint, 1, axis=0)
    #     ori_rf_joint = np.repeat(ori_rf_joint, 1, axis=0)
    #     ori_head_joint = np.repeat(ori_head_joint, 1, axis=0)

    #     # 施加约束
    #     hint[152:153, :3] = ori_root_joint
    #     hint[152:153, 30:33] = ori_lf_joint
    #     hint[152:153, 33:36] = ori_rf_joint

    #     hint[38*3:38*3+1, :3] = ori_root_joint
    #     hint[38*3:38*3+1, 30:33] = ori_lf_joint
    #     hint[38*3:38*3+1, 33:36] = ori_rf_joint

    #     hint[38*2:38*2+1, :3] = ori_root_joint
    #     hint[38*2:38*2+1, 30:33] = ori_lf_joint
    #     hint[38*2:38*2+1, 33:36] = ori_rf_joint
    #     # hint[143:153, 45:48] = ori_head_joint


    
    np.save(f'demo/data/temp_results/hint.npy', hint)

        
