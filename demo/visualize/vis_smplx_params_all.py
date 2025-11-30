import copy
import os
import random
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import trimesh
import open3d as o3d
from scipy.spatial import ConvexHull
import pickle as pkl
from smplx.lbs import vertices2joints
from smplx.utils import Struct, to_tensor, to_np
import sys

sys.path.append('.')
from utils.geometry import compute_twist_rotation, batch_rodrigues, rotation_matrix_to_angle_axis
from utils.rotation_conversions import matrix_to_axis_angle 


from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes

def vis_pointcloud(scene_vert, color, save_name, radius=0.1):
    if scene_vert.shape[0] > 1024:
        indices=np.random.choice(scene_vert.shape[0], 1024, replace=False)
        scene_vert = scene_vert[indices]
    
    # 均匀采样，将数量下降到原来的1/8
    indices = range(0, scene_vert.shape[0], 20)
    scene_vert = scene_vert[indices]

    spheres = []
    for vert in scene_vert:
        vert=vert[[0, 2, 1]]
        vert[2] = -vert[2]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.visual.vertex_colors = color
        sphere.apply_translation(vert)
        spheres.append(sphere)

    # 合并所有小球
    combined_spheres = trimesh.util.concatenate(spheres)

    # 保存mesh
    combined_spheres.export(fr'D:\Documents\我的工作\第四个工作\插图\插图素材\轨迹点\{save_name}_2.ply')

    return combined_spheres

def zup2yup(mesh):
    vertices=mesh.vertices
    vertices[:,[1,2]]=vertices[:,[2,1]]
    vertices[:,2]=-vertices[:,2]
    mesh.vertices=vertices
    return mesh
    
def save_mesh_sequence(obj_mesh, output_dir, is_human=True):
    """
    将Meshes对象的网格序列保存为PLY文件序列
    :param obj_mesh: Meshes类实例
    :param output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取网格数据
    vertices = obj_mesh.vertices  # 形状 (N, V, 3)
    faces = obj_mesh.faces        # 形状 (F, 3)

    # 新增：获取物体变换矩阵
    if not is_human and hasattr(obj_mesh, 'instance_transforms'):
        transforms = obj_mesh.instance_transforms  # 形状 (N, 1, 4, 4)
        rest_vertices = obj_mesh.vertices[0]  # 获取原始未变换的顶点
    else:
        transforms = np.eye(4)
    
    # 遍历每一帧
    for i in tqdm(range(max(vertices.shape[0], transforms.shape[0]))):
        # 创建trimesh对象
        if is_human:
            verts = vertices[i]
        else:
            # 应用变换矩阵
            transform = transforms[i,0]  # 获取当前帧的变换矩阵
            verts = np.einsum('ij,vj->vi', transform[:3,:3], rest_vertices) + transform[:3,3]
            
        mesh = trimesh.Trimesh(
            vertices=verts, 
            faces=faces,
            process=False
        )
        
        # 保存为PLY文件
        if is_human:
            output_path = os.path.join(output_dir, f"{i:05d}.ply")
        else:
            output_path = os.path.join(output_dir, f"{i:05d}_object.ply")
        mesh.export(output_path)

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
    error = (transformation_matrix @ source_points_homogeneous - target_points_homogeneous)[:-1].mean()
    print(f"Error: {error}")

    return R_opt, t_opt, transformation_matrix

human_cup_01 = np.array([
    [0.45064, 1.1269, 0.91], # 00
    [0.47291, 1.1698, 0.91], # 01
    [-1.2303, 0.5421, 0.91], # 02
    [1.5914, 1.5042, 0.91], # 03
    [-0.20433, 0.67891, 0.91], # 04
    [-0.28903, -0.1587, 0.91], # 05
    [0.49608, 0.1642, 0.91], # 06
    [0.31407, 0.20269, 0.91], # 07
    [-1.1036, 1.0028, 0.91], # 08
    [-0.8935, 2.1187, 0.91], # 09
])

human_mouse_01 = [
    [0.45064, -0.33241, 0.91], # 00
    [0.47291, 1.1698, 0.91], # 01
    [0.63463, -1.6846, 0.91], # 02
    [1.9661, 1.0413, 0.91], # 03
    [-0.40178, 0.85005, 0.91], # 04
    [-0.027045, -0.97831, 0.91], # 05
    [0.50155, -1.4363, 0.91], # 06
    [0.31407, 0.20269, 0.91], # 07
    [-1.1036, 1.0028, 0.91], # 08
    [-0.37967, -0.41089, 0.91], # 09
]

human_bottle_03 = [
    [-1.0104655, 0.00820607, 0.91], # 00
    [-0.19351, 3.8911, 0.91], # 01
    [0.57309, -2.5346, 0.91], # 02
    [-0.69809, 1.7775, 0.91], # 03
    [-0.79942, 2.0693, 0.91], # 04
    [0.53408, 2.8951, 0.91], # 05
    [0.52544, 0.1612, 0.91], # 06
    [0.31407, 0.20269, 0.91], # 07
    [-1.1036, 1.0028, 0.91], # 08
    [0.26394, 4.4, 0.91], # 09
]


bottle_03 = np.array([
    [   
        [[-0.371226784, -9.28541577e-01, -1.10229942e-03, -6.00363374e-01],
        [ 9.28361330e-01, -3.71177631e-01,  1.92978729e-02, -1.38723075e+00],
        [-1.83280262e-02,  6.14055513e-03,  9.99813171e-01,  5.27188599e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],

        [[1,0,0,0.7],[0,1,0,-3.5],[0,0,1,0.85774],[0,0,0,1]]
    ],

    [
        [[1,0,0,1.0776],[0,1,0,2.9116],[0,0,1,0.89666],[0,0,0,1]],
        [[1,0,0,-0.28263],[0,1,0,-1.1348],[0,0,1,0.47312],[0,0,0,1]]
    ],

    [
        [[1,0,0,-0.1472],[0,1,0,-2.0532],[0,0,1,0.55449],[0,0,0,1]],
        [[1,0,0,-2.4249],[0,1,0,-2.6553],[0,0,1,0.81195],[0,0,0,1]]
    ],

    [
        [[1,0,0,-0.32369],[0,1,0,0.76516],[0,0,1,0.81543],[0,0,0,1]],
        [[1,0,0,-0.55365],[0,1,0,-2.341],[0,0,1,0.44476],[0,0,0,1]]
    ],

    [
        [[1,0,0,-0.79416],[0,1,0,0.46015],[0,0,1,0.6075],[0,0,0,1]],
        [[1,0,0,1.9252],[0,1,0,-1.7916],[0,0,1,0.46335],[0,0,0,1]]
    ],

    [
        [[1,0,0,1.0223],[0,1,0,0.60384],[0,0,1,1.0816],[0,0,0,1]],
        [[1,0,0,-0.25064],[0,1,0,-2.1563],[0,0,1,0.54433],[0,0,0,1]]
    ],

    [
        [[1,0,0,1.2043],[0,1,0,-0.3214],[0,0,1,1.2074],[0,0,0,1]],
        [[1,0,0,-1.9334],[0,1,0,-2.3129],[0,0,1,0.67124],[0,0,0,1]]
    ], # 06

    [
        [[1,0,0,-1.2757],[0,1,0,0.22572],[0,0,1,0.49781],[0,0,0,1]],
        [[1,0,0,0.59871],[0,1,0,-0.51525],[0,0,1,0.59136],[0,0,0,1]] # 07
    ],

    [
        [[1,0,0,-0.49933],[0,1,0,0.57498],[0,0,1,0.518],[0,0,0,1]],
        [[1,0,0,-0.00068],[0,1,0,0.16909],[0,0,1,0.518],[0,0,0,1]] # 08
    ],

    [
        [[1,0,0,-1.7292],[0,1,0,3.7692],[0,0,1,0.82736],[0,0,0,1]],
        [[1,0,0,-2.0336],[0,1,0,1.1827],[0,0,1,0.73462],[0,0,0,1]] # 09
    ],
])

human_bottle_03 = np.array([
    [-1.0104655, 0.00820607, 0.91], # 00
    [-0.19351, 3.8911, 0.91], # 01
    [0.57309, -2.5346, 0.91], # 02
    [-0.69809, 1.7775, 0.91], # 03
    [-0.79942, 2.0693, 0.91], # 04
    [0.53408, 2.8951, 0.91], # 05
    [0.52544, 0.1612, 0.91], # 06
    [0.31407, 0.20269, 0.91], # 07
    [-1.1036, 1.0028, 0.91], # 08
    [0.26394, 4.4, 0.91], # 09
])

cup_01 = np.array([
    [   
        [[1,0,0,-1.4868],[0,1,0,0],[0,0,1,0.72965],[0,0,0,1]],

        [[-0.00926757, -0.99995706,  0.,   -0.76108     ],
        [ 0.99995706, -0.00926757,  0.,    -1.2829    ],
        [ 0.,          0.,          1.,    0.46016    ],
        [ 0.,          0.,          0.,    1.        ]]
    ],

    [   
        [[1,0,0,-0.72295],[0,1,0,0],[0,0,1,0.4235],[0,0,0,1]],

        [[-1, 0.,  0.,   0.53992    ],
        [ 0, -1.,  0.,   -2.2162    ],
        [ 0., 0.,  1.,    0.46219    ],
        [ 0., 0.,  0.,    1.        ]]
    ],

    [   
        [[-1, 0.,  0.,   -0.36065    ],
        [ 0, -1.,  0.,   -0.52876    ],
        [ 0., 0.,  1.,    0.55872    ],
        [ 0., 0.,  0.,    1.        ]],

        [[-0.70710678, -0.70710678,  0.,   0.88734    ],
        [ 0.70710678, -0.70710678,  0.,   -2.3278    ],
        [ 0., 0.,  1.,    0.50063    ],
        [ 0., 0.,  0.,    1.        ]]
    ],

    [   
        [[-1, 0.,  0.,   3.1005    ],
        [ 0, -1.,  0.,   1.1974    ],
        [ 0., 0.,  1.,   1.1696    ],
        [ 0., 0.,  0.,    1.        ]],

        [[1, 0.,  0.,   0.38137    ],
        [ 0, 1.,  0.,   -2.3783    ],
        [ 0., 0.,  1.,    0.38507  ],
        [ 0., 0.,  0.,    1.       ]]
    ], # 03

    [
         [[-1, 0.,  0.,   1.9211    ],
        [ 0, -1.,  0.,   0.74366    ],
        [ 0., 0.,  1.,   0.40476    ],
        [ 0., 0.,  0.,    1.        ]],

        [[1, 0.,  0.,   -0.88588    ],
        [ 0, 1.,  0.,   0.2898    ],
        [ 0., 0.,  1.,    0.53942  ],
        [ 0., 0.,  0.,    1.       ]]
    ], # 04

    [
         [[1, 0.,  0.,   -1.9161    ],
        [ 0, 1.,  0.,   -1.1039    ],
        [ 0., 0.,  1.,   0.60986    ],
        [ 0., 0.,  0.,    1.        ]],

        [[0., -1,  0.,   -0.28903   ],
        [ 1., 0.,  0.,   -2.1832    ],
        [ 0., 0.,  1.,    0.49029  ],
        [ 0., 0.,  0.,    1.       ]]
    ], # 05

    [
         [[-1, 0.,  0.,   1.182    ],
        [ 0, -1.,  0.,   -0.49312    ],
        [ 0., 0.,  1.,   1.143    ],
        [ 0., 0.,  0.,    1.        ]],

        [[0., -1,  0.,   -0.049129   ],
        [ 1., 0.,  0.,   -2.9759    ],
        [ 0., 0.,  1.,    0.85561  ],
        [ 0., 0.,  0.,    1.       ]]
    ], # 06

    [
         [[-1, 0.,  0.,   0.62008    ],
        [ 0, -1.,  0.,   -1.1441    ],
        [ 0., 0.,  1.,   0.52578    ],
        [ 0., 0.,  0.,    1.        ]],

        [[1., 0.,  0.,   -1.5996   ],
        [ 0., 1.,  0.,   -2.44    ],
        [ 0., 0.,  1.,    0.68111  ],
        [ 0., 0.,  0.,    1.       ]]
    ], # 07

    [
         [[-0.70710678, -0.70710678,  0.,   -0.4568    ],
        [ 0.70710678, -0.70710678,  0.,   0.55275    ],
        [ 0., 0.,  1.,    0.45895    ],
        [ 0., 0.,  0.,    1.        ]],

        [[-0.70710678, -0.70710678,  0.,   -0.86343    ],
        [ 0.70710678, -0.70710678,  0.,   -1.179    ],
        [ 0., 0.,  1.,    0.84396    ],
        [ 0., 0.,  0.,    1.        ]]
    ], # 08

    [
         [[1., 0.,  0.,   -2.0521   ],
        [ 0., 1.,  0.,   1.2211    ],
        [ 0., 0.,  1.,    0.66941  ],
        [ 0., 0.,  0.,    1.       ]],

        [[-1, 0.,  0.,   0.18049    ],
        [ 0, -1.,  0.,   0.62526    ],
        [ 0., 0.,  1.,   0.77379    ],
        [ 0., 0.,  0.,    1.        ]]
    ], # 09
])

human_cup_01 = np.array([
    [0.45064, 1.1269, 0.91], # 00
    [0.47291, 1.1698, 0.91], # 01
    [-1.2303, 0.5421, 0.91], # 02
    [1.5914, 1.5042, 0.91], # 03
    [-0.20433, 0.67891, 0.91], # 04
    [-0.28903, -0.1587, 0.91], # 05
    [0.49608, 0.1642, 0.91], # 06
    [0.31407, 0.20269, 0.91], # 07
    [-1.1036, 1.0028, 0.91], # 08
    [-0.8935, 2.1187, 0.91], # 09

])

mouse_01 = np.array([
    [   
        [[1,0,0,0.93397],[0,1,0,-1.73],[0,0,1,0.47602],[0,0,0,1]],

        [[-0.99837182, -0.05704129,  0., -1.5838     ],
        [ 0.05704129, -0.99837182,  0., -3.4181       ],
        [ 0.,          0.,          1., 0.56719       ],
        [ 0.,          0.,          0.,    1.        ]]
    ],

    [   
        [[-1, 0.,  0.,   -0.70676    ],
        [ 0, -1.,  0.,   -0.0328    ],
        [ 0., 0.,  1.,    0.39529    ],
        [ 0., 0.,  0.,    1.        ]],

        [[-7.37938174e-01, -6.73357934e-01, -4.51258664e-02, -1.5333],
        [ 6.74039759e-01, -7.38695054e-01,  1.44174257e-04, -2.2585],
        [-3.34313352e-02, -3.03102364e-02,  9.98981299e-01, 0.49413],
        [0., 0.,  0.,    1.]]
    ],

    [   
        [[0, 1.,  0.,   0.86423    ],
        [ -1, 0.,  0.,   -2.31    ],
        [ 0., 0.,  1.,    0.49413    ],
        [ 0., 0.,  0.,    1.        ]],

        [[-0.70710678, 0.70710678,  0.,   0.27435    ],
        [ -0.70710678, -0.70710678,  0.,   -4.4076    ],
        [ 0., 0.,  1.,    0.79284    ],
        [ 0., 0.,  0.,    1.        ]]
    ],

    [   
        [[0, 1.,  0.,   1.4981    ],
        [ -1, 0.,  0.,   -1.5589    ],
        [ 0., 0.,  1.,    0.76068    ],
        [ 0., 0.,  0.,    1.        ]],

        [[-1, 0.,  0.,   0.37279    ],
        [ 0, -1.,  0.,   -2.405    ],
        [ 0., 0.,  1.,    0.34847    ],
        [ 0., 0.,  0.,    1.        ]]
    ],# 03

    [   
        [[0, 1.,  0.,   -0.75709    ],
        [ -1, 0.,  0.,   -0.11374    ],
        [ 0., 0.,  1.,    0.51023    ],
        [ 0., 0.,  0.,    1.        ]],

        [[1, 0.,  0.,   0.013278    ],
        [ 0, 1.,  0.,   -2.2442    ],
        [ 0., 0.,  1.,    0.59813    ],
        [ 0., 0.,  0.,    1.        ]]
    ],# 04

    [   
        [[0, 1.,  0.,   -0.40178    ],
        [ -1, 0.,  0.,   -2.1696    ],
        [ 0., 0.,  1.,    0.45792    ],
        [ 0., 0.,  0.,    1.        ]],

        [[-0.52991926, 0.8480481,  0.,   -0.65079    ],
        [ -0.8480481, -0.52991926,  0.,   -3.9039    ],
        [ 0., 0.,  1.,    0.4853    ],
        [ 0., 0.,  0.,    1.        ]]
    ],# 05

    [   
        [[0.25881905, 0.96592583,  0.,   0.68585    ],
        [ -0.96592583, 0.25881905,  0.,   -2.6598    ],
        [ 0., 0.,  1.,    0.46134    ],
        [ 0., 0.,  0.,    1.        ]],

        [[0., 1.,  0.,   -0.76148    ],
        [ -1., 0.,  0.,   -2.961    ],
        [ 0., 0.,  1.,    0.82551    ],
        [ 0., 0.,  0.,    1.        ]]
    ],# 06

    [   
        [[-1., 0.,  0.,   -0.21606    ],
        [ 0, -1.,  0.,   -0.30351    ],
        [ 0., 0.,  1.,    0.41388    ],
        [ 0., 0.,  0.,    1.        ]],

        [[1., 0.,  0.,   0.45444    ],
        [ 0., 1.,  0.,   -2.6854    ],
        [ 0., 0.,  1.,    0.88809    ],
        [ 0., 0.,  0.,    1.        ]]
    ],# 07

    [   
        [[1., 0.,  0.,   -0.46717    ],
        [ 0, 1.,  0.,    0.18288    ],
        [ 0., 0.,  1.,    0.44312    ],
        [ 0., 0.,  0.,    1.        ]],

        [[1., 0.,  0.,   -0.29511    ],
        [ 0., 1.,  0.,   -2.7864    ],
        [ 0., 0.,  1.,    0.44959    ],
        [ 0., 0.,  0.,    1.        ]]
    ],# 08

    [   
       [[-0.70710678, 0.70710678,  0.,   -0.83112    ],
        [ -0.70710678, -0.70710678,  0.,  -0.88754    ],
        [ 0., 0.,  1.,    0.47305    ],
        [ 0., 0.,  0.,    1.        ]],

        [[0., 1.,  0.,   -0.69801    ],
        [ -1., 0.,  0.,   -2.0769    ],
        [ 0., 0.,  1.,    0.82372    ],
        [ 0., 0.,  0.,    1.        ]]
    ],# 09
])

human_mouse_01 = np.array([
    [0.45064, -0.33241, 0.91], # 00
    [0.47291, 1.1698, 0.91], # 01
    [0.63463, -1.6846, 0.91], # 02
    [1.9661, 1.0413, 0.91], # 03
    [-0.40178, 0.85005, 0.91], # 04
    [-0.027045, -0.97831, 0.91], # 05
    [0.50155, -1.4363, 0.91], # 06
    [0.31407, 0.20269, 0.91], # 07
    [-1.1036, 1.0028, 0.91], # 08
    [-0.37967, -0.41089, 0.91], # 09
])

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

smpl_path = fr"data\smplx_models\smplx\SMPLX_NEUTRAL.npz"
smplx_data = np.load(smpl_path, allow_pickle=True)
J_regressor = smplx_data['J_regressor'].astype(np.float32)

def vis_ours(obj_name, motion_id):

    # 读取data并做拼接
    
    data_all=np.load(fr"demo/data/key_frames/data_{obj_name}_{motion_id:02d}/sample_w_scene_guide.npz", allow_pickle=True)
    data_all=dict(data_all)
    if obj_name == 'cup_01':
        human_transl = np.array([human_cup_01[motion_id]])
    elif obj_name == 'mouse_01':
        human_transl = np.array([human_mouse_01[motion_id]])
    elif obj_name == 'bottle_03':
        human_transl = np.array([human_bottle_03[motion_id]])
    human_transl[0, -1] = 0. 
    data_all['transl']+=human_transl
    data_all['obj_transl']+=human_transl

    smpl_squence=SMPLSequence(
        poses_body=data_all["body_pose"][:].reshape(-1, 63),
        smpl_layer=SMPLLayer(model_type="smplx", gender="male"),
        poses_root=data_all["global_orient"][:].reshape(-1, 3),
        trans=data_all["transl"][:],
        poses_right_hand=data_all["right_hand_pose"][:].reshape(-1, 45),
        z_up=True,
    )
    smpl_squence.color = (136/255., 156/255., 216/255., 1.0)
    joints=smpl_squence.joints

    # 初始化
    rest_obj_mesh=trimesh.load_mesh(fr"TRUMAN\object_ply_w_1024\\"+obj_name+".ply")

    # 添加抓取帧及之后的物体轨迹
    obj_trans = data_all['obj_transl'][:, :, None]
    obj_rot = data_all['obj_global_orient']
    obj_trans = np.concatenate([obj_rot, obj_trans], axis=2)
    obj_trans = np.concatenate([obj_trans, np.ones((len(obj_trans), 1, 4))], axis=1)
    obj_trans[:, 3, :3] = 0
    obj_trans = obj_trans.reshape(-1, 1, 4, 4)
    

    object_mesh=Meshes(
        vertices=rest_obj_mesh.vertices,
        faces=rest_obj_mesh.faces,
        instance_transforms=obj_trans,
        z_up=True,
    )
    object_mesh.color = (136/255., 156/255., 216/255., 1.0) 


    rest_scene_mesh=trimesh.load_mesh(fr"TRUMAN\Scene_Mesh_ply\{scene[motion_id]}.ply")
    scene_mesh_ait=Meshes(
        vertices=rest_scene_mesh.vertices,
        faces=rest_scene_mesh.faces,
        z_up=True,
    )

    # 存储human和object的mesh序列
    # save_mesh_sequence(smpl_squence, "scratch\\vis_motions\\results_mesh", is_human=True)
    # save_mesh_sequence(object_mesh, "scratch\\vis_motions\\results_mesh", is_human=False)

    return smpl_squence, object_mesh, scene_mesh_ait ,joints[:, 0]

if __name__ == '__main__':
    obj_name = 'cup_01'
    motion_id = 2

    smpl_squence0, object_mesh0, scene_mesh_ait0, root_joints0 = vis_ours(obj_name, motion_id)

    root_joints0[:, 2] = 0.91

    # 可视化
    v = Viewer()
    v.scene.add(smpl_squence0)
    v.scene.add(scene_mesh_ait0)
    v.scene.add(object_mesh0)
    v.run()