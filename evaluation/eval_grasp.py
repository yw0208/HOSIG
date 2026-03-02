import os
import torch
import numpy as np
from tqdm import tqdm
import trimesh
import pickle as pkl
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points


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


if __name__ == '__main__':
    methods = ['ours']
    for method in methods:
        obj_name_list = ['bottle_03', 'cup_01', 'mouse_01']

        nn_dist_list_scene = []
        interior_list_scene = []
        nn_dist_list_obj = []
        interior_list_obj = []
        for motion_id in tqdm(range(9), desc=method.split('/')[-1]):
            for obj_name in obj_name_list:
                # 1. 读取所有的mesh
                scene_mesh = trimesh.load(os.path.join('TRUMANS/Scene_Mesh_ply', scene[motion_id] + '.ply'))
                human_mesh_list = []
                obj_mesh_list = []
                mesh_dir = f"sgap\saves\GNet_3obj\sample_results_E120\{obj_name}_grasp_{motion_id:02d}"
                for person_id in range(2):
                    for frame_id in range(10):
                        human_mesh = trimesh.load(os.path.join(mesh_dir, f"{person_id:04d}_{frame_id:04d}_sbj_refine.ply"))
                        obj_mesh = trimesh.load(os.path.join(mesh_dir, f"{person_id:04d}_{frame_id:04d}_obj.ply"))
                        human_mesh_list.append(human_mesh)
                        obj_mesh_list.append(obj_mesh)
                
                # 2. 计算场景交互
                for frame_id in range(len(human_mesh_list)):
                    human_mesh = human_mesh_list[frame_id]
                    nn_dist, interior = cal_sdf(human_mesh, scene_mesh)
                    nn_dist_list_scene.append(nn_dist)
                    interior_list_scene.append(interior)
                
                # 3. 计算物体交互                
                for frame_id in range(2):
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
        with open(f"evaluation/eval_results/penetration_{method.split('/')[-1]}.pkl", 'wb') as f:
            pkl.dump({
                'scene_pene_rate': scene_pene_rate,
                'scene_pene_mean': scene_pene_mean,
                'scene_pene_max': scene_pene_max,
                'obj_pene_rate': obj_pene_rate,
                'obj_pene_mean': obj_pene_mean,
                'obj_pene_max': obj_pene_max
            }, f)

    
    
        


            

                
                


        