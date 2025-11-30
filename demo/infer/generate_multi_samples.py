import copy
import os
import joblib
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append('.')
from demo.demo_utils.vis_mdm_rw_rh import joints2smplx_obj_motion
from demo.demo_utils.get_hint import get_hint_for_generate, rotation_matrix_zup2yup
from sample.generate import main


def simplify_scene(sparse_path, scene_pointcloud):
    # 确定sparse_path范围
    min_x = np.min(sparse_path[:, 0])
    max_x = np.max(sparse_path[:, 0])
    min_y = np.min(sparse_path[:, 1])
    max_y = np.max(sparse_path[:, 1])
    min_z = 0.2
    max_z = 1.8

    # 去掉scene_pointcloud中超出范围的点
    new_scene_pointcloud = []
    for i in range(len(scene_pointcloud)):
        if min_x-0.8 < scene_pointcloud[i][0] < max_x+0.8 and min_y-0.8 < scene_pointcloud[i][1] < max_y+0.8 and min_z < scene_pointcloud[i][2] < max_z:
            new_scene_pointcloud.append(scene_pointcloud[i])
    scene_pointcloud = np.array(new_scene_pointcloud)

    return scene_pointcloud


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

human_bottle_03 = np.array([
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

if __name__=='__main__':
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

    # 
    for obj_name in ['bottle_03', 'cup_01', 'mouse_01']:
        for sample_id in tqdm(range(10), desc="Generating samples"):
            grasp_fullpose_rotmat_path = []
            grasp_transl_path = []
            key_frames_dir = f"demo/data/key_frames/data_{obj_name}_{sample_id:02d}/"
            for file_id, file_name in enumerate(sorted(os.listdir(key_frames_dir))):
                if file_id == 0:
                    grasp_fullpose_rotmat_path.append(os.path.join(key_frames_dir, file_name))
                elif file_id == 1:
                    grasp_transl_path.append(os.path.join(key_frames_dir, file_name))
                elif file_id == 2:
                    grasp_fullpose_rotmat_path.append(os.path.join(key_frames_dir, file_name))
                elif file_id == 3:
                    grasp_transl_path.append(os.path.join(key_frames_dir, file_name))
                elif file_id == 5:
                    path_0 = np.load(os.path.join(key_frames_dir, file_name))
                    # 预防第0帧就抓了
                    if path_0[0, -1] != 0.91:
                        path_0[1, -1] = path_0[0, -1]
                        path_0[0, -1] = 0.91
                elif file_id == 6:
                    path_1 = np.load(os.path.join(key_frames_dir, file_name))
                                
            scene_name = scene[sample_id]
            scene_pointcloud = np.load(f"TRUMAN/Scene/{scene_name}.npy")
            # 预处理，需要得到一个n*3的点云
            coords = np.argwhere(scene_pointcloud)
            coords = coords / 50.
            coords[:, 0]=coords[:, 0]-3.
            coords[:, 2]=coords[:, 2]-4.
            scene_pointcloud = coords
            print("scene_pointcloud shape:", scene_pointcloud.shape)

            # 加载human和obj translation
            if obj_name == 'bottle_03':
                human_transl = [human_bottle_03[sample_id]]
            elif obj_name == 'cup_01':
                human_transl = [human_cup_01[sample_id]]
            elif obj_name == 'mouse_01':
                human_transl = [human_mouse_01[sample_id]]
            human_transl[0][2] = 0.

            if obj_name == 'bottle_03':
                obj_transl = bottle_03[sample_id, :, :3, 3].reshape(-1, 3)
            elif obj_name == 'cup_01':
                obj_transl = cup_01[sample_id, :, :3, 3].reshape(-1, 3)
            elif obj_name == 'mouse_01':
                obj_transl = mouse_01[sample_id, :, :3, 3].reshape(-1, 3)
            obj_transl[:, 2]= 0.

            # 加载整个路径
            sparse_path = np.concatenate([path_0, path_1[1:]], axis=0)
            sparse_path = (rotation_matrix_zup2yup@sparse_path.T).T #转成y-up坐标

            # 对scene_pointcloud进行简化，只保留sparse_path周围0.8米内的点

            # 去掉scene_pointcloud中超出0.8范围的点
            new_scene_pointcloud = []
            dist_scene_path = torch.cdist(torch.from_numpy(scene_pointcloud), torch.from_numpy(sparse_path))
            for i in range(len(scene_pointcloud)):
                if torch.min(dist_scene_path[i]) < 0.8:
                    new_scene_pointcloud.append(scene_pointcloud[i])
            scene_pointcloud = np.array(new_scene_pointcloud)
            print("new scene_pointcloud shape:", scene_pointcloud.shape)

            grasp_frame_id = 0
            put_frame_id = 0

            all_resluts_dict_list = []
            # 将路径分为多段动作
            grasp_time = 0
            last_motion_node = np.array([human_transl[0][0], 0.0, -human_transl[0][1]]) # y坐标
            for start_index in tqdm(range(0, len(sparse_path)-4, 4)):
                trajectory = copy.deepcopy(sparse_path[start_index:start_index+5])

                # # 如果不是第1段动作，需要将这段动作的起点对齐到第1段动作的终点
                if start_index != 0:
                    trajectory[0] = last_motion_node
                    trajectory[0][1] = 0.91
                
                simple_scene = simplify_scene(trajectory, scene_pointcloud)
                print("simple_scene shape:", simple_scene.shape)

                t=np.array([[trajectory[0, 0], 0, trajectory[0, 2]]]) # 用于对齐起点
                # 判断是否涉及抓住物体
                grasp_flag = False
                grasp_index = 0
                for i in range(4, -1, -1):
                    if trajectory[i, 1] == 0.91:
                        continue
                    else: # 避免开始和结束抓住
                        grasp_flag = True
                        grasp_time += 1
                        grasp_index=i

                        # 计算第几帧抓住和放下
                        if grasp_time == 1:
                            grasp_frame_id = start_index // 4 * 153 + i * 38
                        elif grasp_time == 2:
                            put_frame_id = 153 + (start_index // 4 - 1) * 152 + i * 38
                        break
            
                get_hint_for_generate(trajectory, t, grasp_flag, grasp_time, grasp_index, start_index, obj_transl, grasp_fullpose_rotmat_path, grasp_transl_path, sparse_path) # 得到hint
                # 根据不同阶段来选择不同的prompt
                if grasp_time == 0 and grasp_index==0 and start_index==0:
                    # 说明还没抓到, 起立
                    results_dict = main(0, simple_scene, last_motion_node) 
                elif grasp_time == 0 and grasp_index==0 and start_index==4:
                    # 说明还没抓到, 起立
                    results_dict = main(4, simple_scene, last_motion_node) 
                elif grasp_time == 1 and grasp_index != 0:
                    # 抓取帧
                    results_dict = main(1, simple_scene, last_motion_node) 
                elif grasp_time == 1 and grasp_index == 0:
                    # 抓到了，还在走
                    results_dict = main(2, simple_scene, last_motion_node)
                elif grasp_time == 2 and grasp_index != 0:
                    # 放下帧
                    results_dict = main(3, simple_scene, last_motion_node)
                elif grasp_time == 2 and grasp_index == 0 and start_index < len(sparse_path)-8:
                    # 放下后，还在走
                    results_dict = main(4, simple_scene, last_motion_node)
                elif start_index >= len(sparse_path)-8:
                    # 最后一段, 坐下
                    results_dict = main(5, simple_scene, last_motion_node)

                all_resluts_dict_list.append(results_dict)

                # 更新last_motion_node
                joints = copy.deepcopy(results_dict['motion'][0].transpose(2, 0, 1)[:153])
                joints[..., 1]=0.
                last_motion_node += joints[-1, 0, :]

            joblib.dump(all_resluts_dict_list, "demo/data/temp_results/all_resluts_dict_list.pkl")
            all_resluts_dict_list = joblib.load("demo/data/temp_results/all_resluts_dict_list.pkl")
            joints2smplx_obj_motion(all_resluts_dict_list=all_resluts_dict_list, grasp_index=grasp_frame_id, put_index=put_frame_id-1, human_init_transl=human_transl, obj_start_end_transl=obj_transl, sparse_path=sparse_path, grasp_fullpose_rotmat_path=grasp_fullpose_rotmat_path, grasp_transl_path=grasp_transl_path) # joint转smplx