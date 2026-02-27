import math
import os
import numpy as np
import torch
from tqdm import tqdm
import trimesh
from bps_torch.bps import bps_torch
import pytorch3d.transforms as t3d

import sys
sys.path.append('.')

bps = torch.load("bps_trumans_15_scene.pt")
bps_torch = bps_torch()
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def rotmat2aa(pose):
    pose = to_tensor(pose)
    reshaped_input = pose.reshape(-1, 3, 3)
    quat = t3d.matrix_to_quaternion(reshaped_input)
    return t3d.quaternion_to_axis_angle(quat)

if __name__ == '__main__':
    object_info={}
    obj_dir="/opt/data/private/datasets/TRUMANS/processed_data/captured_objects"
    for obj_file in tqdm(sorted(os.listdir(obj_dir)),desc='OBJ_INF'):
        obj_name=obj_file.replace('.ply','')
        if obj_name in ['cup_01', 'mouse_01', 'bottle_03']:
            obj_mesh=trimesh.load_mesh(os.path.join(obj_dir, obj_file))
            obj_verts=obj_mesh.vertices
            obj_center=np.mean(obj_verts, axis=0)

            object_info[obj_name]={}
            object_info[obj_name]['center']=obj_center
            object_info[obj_name]['verts']=obj_verts
    
    obj_info = np.load("/opt/data/private/datasets/trumans4gnet/obj_info.npy", allow_pickle=True).item()

    obj_transformation_list_list = [
        np.array([[-1, 0.,  0.,   -0.70676    ],
        [ 0, -1.,  0.,   -0.0328    ],
        [ 0., 0.,  1.,    0.39529    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[-7.37938174e-01, -6.73357934e-01, -4.51258664e-02, -1.5333],
        [ 6.74039759e-01, -7.38695054e-01,  1.44174257e-04, -2.2585],
        [-3.34313352e-02, -3.03102364e-02,  9.98981299e-01, 0.49413],
        [0., 0.,  0.,    1.]]),

        np.array([[0, 1.,  0.,   0.86423    ],
        [ -1, 0.,  0.,   -2.31    ],
        [ 0., 0.,  1.,    0.49413    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[-0.70710678, 0.70710678,  0.,   0.27435    ],
        [ -0.70710678, -0.70710678,  0.,   -4.4076    ],
        [ 0., 0.,  1.,    0.79284    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[0, 1.,  0.,   1.4981    ],
        [ -1, 0.,  0.,   -1.5589    ],
        [ 0., 0.,  1.,    0.76068    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[-1, 0.,  0.,   0.37279    ],
        [ 0, -1.,  0.,   -2.405    ],
        [ 0., 0.,  1.,    0.34847    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[0, 1.,  0.,   -0.75709    ],
        [ -1, 0.,  0.,   -0.11374    ],
        [ 0., 0.,  1.,    0.51023    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[1, 0.,  0.,   0.013278    ],
        [ 0, 1.,  0.,   -2.2442    ],
        [ 0., 0.,  1.,    0.59813    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[0, 1.,  0.,   -0.40178    ],
        [ -1, 0.,  0.,   -2.1696    ],
        [ 0., 0.,  1.,    0.45792    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[-0.52991926, 0.8480481,  0.,   -0.65079    ],
        [ -0.8480481, -0.52991926,  0.,   -3.9039    ],
        [ 0., 0.,  1.,    0.4853    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[0.25881905, 0.96592583,  0.,   0.68585    ],
        [ -0.96592583, 0.25881905,  0.,   -2.6598    ],
        [ 0., 0.,  1.,    0.46134    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[0., 1.,  0.,   -0.76148    ],
        [ -1., 0.,  0.,   -2.961    ],
        [ 0., 0.,  1.,    0.82551    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[-1., 0.,  0.,   -0.21606    ],
        [ 0, -1.,  0.,   -0.30351    ],
        [ 0., 0.,  1.,    0.41388    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[1., 0.,  0.,   0.45444    ],
        [ 0., 1.,  0.,   -2.6854    ],
        [ 0., 0.,  1.,    0.88809    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[1., 0.,  0.,   -0.46717    ],
        [ 0, 1.,  0.,    0.18288    ],
        [ 0., 0.,  1.,    0.44312    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[1., 0.,  0.,   -0.29511    ],
        [ 0., 1.,  0.,   -2.7864    ],
        [ 0., 0.,  1.,    0.44959    ],
        [ 0., 0.,  0.,    1.        ]]),

       np.array([[-0.70710678, 0.70710678,  0.,   -0.83112    ],
        [ -0.70710678, -0.70710678,  0.,  -0.88754    ],
        [ 0., 0.,  1.,    0.47305    ],
        [ 0., 0.,  0.,    1.        ]]),

        np.array([[0., 1.,  0.,   -0.69801    ],
        [ -1., 0.,  0.,   -2.0769    ],
        [ 0., 0.,  1.,    0.82372    ],
        [ 0., 0.,  0.,    1.        ]])
]

    obj_name = 'mouse_01'
    scene_dir = "/opt/data/private/datasets/TRUMANS/Scene"
    scene = [
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
    
    for scene_id in range(len(scene)):
        scene_np=np.load(os.path.join(scene_dir, scene[scene_id]+'.npy'))
        obj_transformation_list = obj_transformation_list_list[scene_id*2: scene_id*2+2]
        batch = {
            'transl_obj':[],
            'betas':[],
            'bps_obj_glob':[],
            'scene_bps':[],
            'verts_obj':[],
            'global_orient_obj':[],
            'scene_verts':[],
        }
        for idx, obj_transformation in enumerate(obj_transformation_list):
            obj_trans = obj_transformation[:3, -1]
            obj_trans=np.array(obj_trans)
            global_orient_rotmat_obj=obj_transformation[:3, :3]
            global_orient_obj=rotmat2aa(global_orient_rotmat_obj)
            batch['global_orient_obj'].append(global_orient_obj)
            verts_obj=torch.from_numpy(((global_orient_rotmat_obj@object_info[obj_name]['verts'].T).T+obj_trans[None,:])).unsqueeze(0)

            sample_verts_obj = obj_info[obj_name]['verts_sample']
            sample_obj_trans = obj_trans.copy()
            sample_obj_trans[:2]=np.zeros(2)
            sample_verts_obj = torch.from_numpy(((global_orient_rotmat_obj@sample_verts_obj.T).T+sample_obj_trans[None,:])).unsqueeze(0)
            batch['verts_obj'].append(to_cpu(sample_verts_obj))
            
            trans_obj=obj_trans
            
            # get the bps_obj_glob
            obj_bps = bps['obj'] + torch.from_numpy(obj_trans[None, None, :])
            bps_obj_glob = bps_torch.encode(x=verts_obj,
                                        feature_type=['deltas'],
                                        custom_basis=obj_bps)['deltas']
            bps_obj_glob=to_cpu(bps_obj_glob)

            trans=trans_obj
            scene_range=[50.*(np.array([trans[0]-0.8, trans[0]+0.8])+3.), 50.*np.array([0.2, 1.8]), 50.*(np.array([trans[1]-0.8, trans[1]+0.8])-4.)*-1.]
            scene_seg=scene_np[math.floor(scene_range[0][0]):math.ceil(scene_range[0][1]), math.floor(scene_range[1][0]):math.ceil(scene_range[1][1]), math.floor(scene_range[2][1]):math.ceil(scene_range[2][0])]
            coords = np.argwhere(scene_seg)
            coords+=np.array([[math.floor(scene_range[0][0]), math.floor(scene_range[1][0]), math.floor(scene_range[2][1])]])
            coords=coords/50.
            coords=coords[:,[0, 2, 1]]
            coords[:, 1]=coords[:, 1]*-1.
            coords[:, 0]=coords[:, 0]-3.
            coords[:, 1]=coords[:, 1]+4.
            scene_verts=np.array(coords, dtype=np.float32)

            # canonolize scene
            scene_verts[:,:2]=scene_verts[:,:2] - trans[None,:2]
            if scene_verts.shape[0] <= 0:
                scene_verts=np.zeros((1, 3))

            batch['scene_verts'].append(scene_verts)

            indices=np.random.choice(scene_verts.shape[0], 1024, replace=True)
            scene_verts = scene_verts[indices]

            # get bps
            obj_bps = bps['scene'].repeat(1, 1, 1) + torch.tensor([[[0., 0., 1.]]], dtype=torch.float32)
            if scene_verts.shape[0] > 0:
                scene_bps = bps_torch.encode(x=torch.from_numpy(scene_verts).float(),
                                            feature_type=['deltas'],
                                            custom_basis=obj_bps)['deltas']
                # plot_3dpoint(scene_verts, obj_bps)
                batch['scene_bps'].append(to_cpu(scene_bps))
            else:
                batch['scene_bps'].append(np.zeros((1, 1024, 3)))

            trans_obj[:2] = trans_obj[:2]-trans[:2]
            batch['transl_obj'].append(trans_obj)
            batch['betas'].append(np.zeros((1,10)))
            batch['bps_obj_glob'].append(bps_obj_glob)
        np.save(f'generate/multi_samples/batches/batch_{scene_id}.npy', batch)
