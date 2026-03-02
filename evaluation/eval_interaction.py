import os
import torch
import numpy as np
from tqdm import tqdm
import trimesh
import pickle as pkl
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points


def vis_pointcloud(scene_vert):
    if scene_vert.shape[0] > 1024:
        indices=np.random.choice(scene_vert.shape[0], 1024, replace=False)
        scene_vert = scene_vert[indices]
    spheres = []
    for vert in scene_vert:
        vert=vert[[0, 2, 1]]
        vert[2] = -vert[2]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.06)
        sphere.apply_translation(vert)
        spheres.append(sphere)

    # 合并所有小球
    combined_spheres = trimesh.util.concatenate(spheres)

    return combined_spheres

def zup2yup(mesh):
    vertices=mesh.vertices
    vertices[:,[1,2]]=vertices[:,[2,1]]
    vertices[:,2]=-vertices[:,2]
    mesh.vertices=vertices
    return mesh


def transform_points(points, transformation_matrix):
    """应用旋转矩阵和平移向量到点集"""
    seq_len = points.shape[0]
    points=points.reshape(-1,3)
    points=points.T

    points_homogeneous = np.vstack((points, np.ones((1,points.shape[1]))))
    transformed_points = transformation_matrix @ points_homogeneous
    transformed_points = transformed_points[:3].T

    transformed_points = transformed_points.reshape(seq_len, -1, 3)

    return transformed_points

def estimate_transform(source_points, target_points):
    """估计两个三维点集之间的变换矩阵"""
    source_points = source_points.T
    target_points = target_points.T

    # 将点集转换为齐次坐标
    source_points_homogeneous = np.vstack((source_points, np.ones((1,source_points.shape[1]))))
    target_points_homogeneous = np.vstack((target_points, np.ones((1,source_points.shape[1]))))
    
    transformation_matrix=target_points_homogeneous @ np.linalg.pinv(source_points_homogeneous)
    R_opt = transformation_matrix[:3, :3]
    t_opt = transformation_matrix[:3, 3:]
    
    # compute the error
    # error = (transformation_matrix @ source_points_homogeneous - target_points_homogeneous)[:-1].mean()
    # print(f"Error: {error}")

    return R_opt, t_opt, transformation_matrix


scene = [
    '4a3101c0-29f7-4e2b-9e24-10bd984ac084', # 00
    '5a2a7c8d-8a68-47b2-9193-69d73c9e2d95', # 01
    'a2e8ba09-af97-4301-a9ef-f5e583c853dd', # 02
    '0a761819-05d1-4647-889b-a726747201b1', # 03
    '1d19e06d-bbe7-4d3d-a65b-60b3fe01b8a2', # 04
    '2a6c3151-0e15-42e4-878a-e890e9a9d946', # 05
    '2b4c9b84-eede-4ef8-b850-d3cd58fa61f7', # 06
    '3a3f479b-3fc5-4bf5-a60b-637492422f45', # 07
    '00add26c-7a26-4a61-b192-b97aa493b3f3', # 08
    '1a1e205b-3f54-49fb-8154-e2d61c3682ae', # 09
]




def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx 

def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)

def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    '''
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
    return interior

def cal_sdf(human_mesh, occ_mesh):
    # build mesh
    occ_mesh_vertices = occ_mesh.vertices
    occ_mesh_vertices_tensor = torch.tensor(occ_mesh_vertices, dtype=torch.float32).view(1, -1, 3).to('cuda')
    occ_mesh_faces = occ_mesh.faces
    occ_mesh_faces_tensor = torch.tensor(occ_mesh_faces, dtype=torch.int64).view(1, -1, 3).to('cuda')
    occ_meshes = Meshes(verts=occ_mesh_vertices_tensor, faces=occ_mesh_faces_tensor)

    # get normal
    occ_normals = occ_meshes.verts_normals_packed().view(1, -1, 3)

    # get scene and human verts distance
    human_mesh_vertices = human_mesh.vertices
    # 只保留z高度大于0.2的点
    human_mesh_vertices = human_mesh_vertices[human_mesh_vertices[:, 2] > 0.2]
    human_mesh_vertices_tensor = torch.tensor(human_mesh_vertices, dtype=torch.float32).view(1, -1, 3).to('cuda')
    nn_dist, nn_idx = get_NN(human_mesh_vertices_tensor, occ_mesh_vertices_tensor)

    # get scene verts inside human
    interior = get_interior(occ_normals,occ_mesh_vertices_tensor, human_mesh_vertices_tensor, nn_idx).type(torch.bool)

    # caculate inside distance

    return nn_dist, interior

def compute_sdf(mesh, points):
    """
    计算点云与网格之间的符号距离场（SDF）
    
    参数:
        mesh (trimesh.Trimesh): 输入的三角形网格（需为水密网格）
        points (np.ndarray): 点云坐标，形状为 (N, 3)
        
    返回:
        np.ndarray: 每个点的 SDF 值，形状为 (N,)
    """
    # 检查网格是否水密
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight to determine inside/outside.")
    
    # 计算每个点到网格表面的最近距离（始终为非负数）
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh, points)
    
    # 判断点是否在网格内部（水密条件下）
    inside = mesh.contains(points)
    
    # 符号约定：内部为负，外部为正
    sdf = distances * np.where(inside, -1, 1)
    
    return sdf

def get_key_frames(method, obj_name, motion_id):
    if method == 'ours':
        data_dir = f"demo/data/key_frames/data_{obj_name}_{motion_id:02d}"
        file_list = sorted(os.listdir(data_dir))
        path_1 = np.load(os.path.join(data_dir, file_list[5]))
        path_2 = np.load(os.path.join(data_dir, file_list[6]))
        path = np.concatenate((path_1, path_2[1:]), axis=0)
        path = list(path)

        pick_frame = 0
        put_frame = 0
        for index, root in enumerate(path):
            if root[-1] != 0.91 and pick_frame == 0 and put_frame == 0:
                pick_frame = index
            elif root[-1] != 0.91 and pick_frame != 0 and put_frame == 0:
                put_frame = index
                break
        pick_frame = pick_frame * 38
        put_frame = put_frame * 38
        return pick_frame, put_frame



if __name__ == '__main__':
    methods = ['ours']
    for method in methods:
        obj_name_list = ['bottle_03', 'cup_01', 'mouse_01']

        nn_dist_list_scene = []
        interior_list_scene = []
        nn_dist_list_obj = []
        interior_list_obj = []
        for motion_id in tqdm(range(10), desc=method):
            for obj_name in obj_name_list:
                # 1. 读取所有的mesh
                scene_mesh = trimesh.load(os.path.join('TRUMANS/Scene_Mesh_ply', scene[motion_id] + '.ply'))
                human_mesh_list = []
                obj_mesh_list = []
                mesh_dir = f"evaluation/{method}/scene_{motion_id}/{obj_name}/"
                for frame_id in range(len(os.listdir(mesh_dir))//2):
                    human_mesh = trimesh.load(os.path.join(mesh_dir, f"human_{frame_id:03d}.ply"))
                    obj_mesh = trimesh.load(os.path.join(mesh_dir, f"obj_{frame_id:03d}.ply"))
                    human_mesh_list.append(human_mesh)
                    obj_mesh_list.append(obj_mesh)
                
                # 2. 计算场景交互
                for frame_id in range(len(human_mesh_list)):
                    human_mesh = human_mesh_list[frame_id]
                    nn_dist, interior = cal_sdf(human_mesh, scene_mesh)
                    nn_dist_list_scene.append(nn_dist)
                    interior_list_scene.append(interior)
                
                # 3. 计算物体交互
                # 3.1 确定拿起帧和放下帧
                pick_frame, put_frame = get_key_frames(method, obj_name, motion_id)
                if put_frame == -120:
                    put_frame = len(human_mesh_list) - 120

                if put_frame>=len(human_mesh_list):
                    pick_frame = 76
                    put_frame = 495

                for frame_id in range(pick_frame, put_frame-1):
                    human_mesh = human_mesh_list[frame_id]
                    obj_mesh = obj_mesh_list[frame_id]
                    nn_dist, interior = cal_sdf(human_mesh, obj_mesh)
                    nn_dist_list_obj.append(nn_dist)
                    interior_list_obj.append(interior)
                
        
        # 4. 统计场景相关结果
        scene_pene_count = 0
        scene_pene_list = []
        for i in range(len(nn_dist_list_scene)):
            nn_dist = nn_dist_list_scene[i]
            interior = interior_list_scene[i]
            penetr_dist = torch.sum(nn_dist[interior])
            if penetr_dist > 0.001:
                scene_pene_count += 1
            
            scene_pene_list.append(penetr_dist)

        scene_pene_rate = scene_pene_count / len(nn_dist_list_scene)
        scene_pene_mean = torch.mean(torch.tensor(scene_pene_list))
        scene_pene_max = torch.max(torch.tensor(scene_pene_list))

        # 5. 统计物体相关结果
        obj_pene_count = 0
        obj_pene_list = []
        for i in range(len(nn_dist_list_obj)):
            nn_dist = nn_dist_list_obj[i]
            interior = interior_list_obj[i]
            penetr_dist = torch.sum(nn_dist[interior])
            if penetr_dist > 0.:
                obj_pene_count += 1
            obj_pene_list.append(penetr_dist)
        
        obj_pene_rate = obj_pene_count / len(nn_dist_list_obj)
        obj_pene_mean = torch.mean(torch.tensor(obj_pene_list))
        obj_pene_max = torch.max(torch.tensor(obj_pene_list))

        # 6. 打印结果
        print(f"Scene Penetration Rate: {scene_pene_rate:.4f}")
        print(f"Scene Penetration Mean: {scene_pene_mean:.4f}")
        print(f"Scene Penetration Max: {scene_pene_max:.4f}")
        print(f"Object Penetration Rate: {obj_pene_rate:.4f}")
        print(f"Object Penetration Mean: {obj_pene_mean:.4f}")
        print(f"Object Penetration Max: {obj_pene_max:.4f}")

        # 7. 保存结果
        with open(f"scratch/evaluation/eval_results/penetration_{method}.pkl", 'wb') as f:
            pkl.dump({
                'scene_pene_rate': scene_pene_rate,
                'scene_pene_mean': scene_pene_mean,
                'scene_pene_max': scene_pene_max,
                'obj_pene_rate': obj_pene_rate,
                'obj_pene_mean': obj_pene_mean,
                'obj_pene_max': obj_pene_max
            }, f)

    
    
        


            

                
                


        