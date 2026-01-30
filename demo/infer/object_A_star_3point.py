import os
import random
import numpy as np
from tqdm import tqdm
import trimesh
import heapq
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from demo.data.obj_set import *

def heuristic(array, a, b, ori_array):
    if a[0]!=b[0]: 
        A=float((a[1]-b[1])/(a[0]-b[0]))
        B=b[1]-A*b[0]

        if a[0]>b[0]:
            x1=b[0]
            x2=a[0]
        else:
            x1=a[0]
            x2=b[0]
        
        path_cost=0
        for x in range(np.round(x1),np.round(x2)+1):
            y=round(A*x+B)
            path_cost+=array[x][y]
    else:
        if a[1]>b[1]:
            y1=b[1]
            y2=a[1]
        else:
            y1=a[1]
            y2=b[1]

        path_cost=0
        for y in range(np.round(y1),np.round(y2)+1):
            path_cost+=array[a[0]][y]
    
    # 加上a到b的直线距离
    path_cost += np.linalg.norm(np.array(a)-np.array(b))*7.5

    # 防止和场景穿模，假如该点有场景，额外施加大惩罚
    if ori_array[a[0]][a[1]]>0:
        path_cost += 1000000

    return path_cost

def astar(array, ori_array, start, goal):
    neighbors = [(1,0),(0,-1),(0,1),(-1,0)] # 上下左右四个方向
    close_set = set()
    came_from = {}
    gscore = {start:array[start[0]][start[1]]}
    fscore = {start:heuristic(array, start, goal, ori_array)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j  

            # 避免进入家具部分
            # if ori_array[neighbor[0]][neighbor[1]]>0:
            #     continue

            # try:
            #     # 避免进入家具部分
            #     if ori_array[neighbor[0]][neighbor[1]]>0:
            #         continue
            # except:
            #     data = []
            #     while current in came_from:
            #         data.append(current)
            #         current = came_from[current]
            #     # visualize the smoothed scene_2d_map
            #     plt.imshow(array, cmap='viridis')
            #     plt.colorbar()
            #     plt.show()
            #     # visualize the path
            #     for point in data:
            #         plt.scatter(point[1], point[0], c='r', marker='o', s=180)
            #     plt.imshow(array, cmap='viridis')
            #     plt.colorbar()
            #     plt.show()

            try:
                tentative_g_score = gscore[current] + array[neighbor[0]][neighbor[1]]
            except:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                # visualize the smoothed scene_2d_map
                plt.imshow(ori_array, cmap='viridis')
                plt.colorbar()
                plt.show()
                # visualize the path
                for point in data:
                    plt.scatter(point[1], point[0], c='r', marker='o', s=180)
                plt.scatter(goal[1], goal[0], c='r', marker='o', s=180)
                plt.imshow(ori_array, cmap='viridis')
                plt.colorbar()
                plt.show()
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(array, neighbor, goal, ori_array)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return False

def convolve_2d(arr):
    # 获取数组的大小
    rows, cols = arr.shape
    
    # 创建一个与输入数组相同大小的零数组
    result = np.zeros_like(arr)
    
    # # 定义卷积核，即周围8个位置的权重
    # kernel = np.ones((3, 3)) / 9
    
    # # 遍历数组中的每个元素，除了边界元素
    # for i in range(1, rows-1):
    #     for j in range(1, cols-1):
    #         # 使用卷积核对当前元素及其周围的元素进行卷积操作
    #         result[i, j] = np.sum(kernel * arr[i-1:i+2, j-1:j+2])

    # 定义卷积核，即周围8个位置的权重
    kernel = np.ones((5, 5)) / 25
    
    # 遍历数组中的每个元素，除了边界元素
    for i in range(2, rows-2):
        for j in range(2, cols-2):
            # 使用卷积核对当前元素及其周围的元素进行卷积操作
            result[i, j] = np.sum(kernel * arr[i-2:i+3, j-2:j+3])
    
    return result

def random_move(scene_2d_map, node):# 保险机制防止node在scene_2d_map上的值大于0
    if scene_2d_map[node[0]][node[1]]==0:
        return node
    
    # 先找node周围8个点中为0的点，如果存在为0的点，则随机选择一个点作为新的node
    # neighbors = [(1,0),(0,-1),(0,1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    # neighbors = [(node[0]+i, node[1]+j) for i, j in neighbors]
    # new_neighbors = []

    # for neighbor in neighbors:
    #     if scene_2d_map[neighbor[0]][neighbor[1]]==0:
    #         new_neighbors.append(neighbor)

    # if len(new_neighbors)>0:
    #     return random.choice(new_neighbors)
    # else:
    #     # 找node更外一层16个点中为0的点，如果存在为0的点，则随机选择一个点作为新的node
    #     neighbors = [(2,0),(0,-2),(0,2),(-2,0),(2,2),(2,-2),(-2,2),(-2,-2),(2,1),(2,-1),(-2,1),(-2,-1),(1,-2),(-1,2),(-1,-2),(1,2)]
    #     neighbors = [(node[0]+i, node[1]+j) for i, j in neighbors]
    #     new_neighbors = []

    #     for neighbor in neighbors:
    #         if scene_2d_map[neighbor[0]][neighbor[1]]==0:
    #             new_neighbors.append(neighbor)

    #     if len(new_neighbors)>0:
    #         return random.choice(new_neighbors)
    count=1
    while True:
        neighbors = []
        for i in range(-count,count+1):
            for j in range(-count,count+1):
                neighbors.append((node[0]+i, node[1]+j))
        new_neighbors = []
        for neighbor in neighbors:
            if scene_2d_map[neighbor[0]][neighbor[1]]==0:
                new_neighbors.append(neighbor)
        if len(new_neighbors)>0:
            return random.choice(new_neighbors)
        else:
            count+=1
    
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

if __name__=='__main__':
    # 如果想更改物品，直接用replace，例如ctrl+h，将cup_01替换为chair_01
    scene_id = 1

    #  convert scene to a 2d array
    scene_mesh=trimesh.load_mesh(f'E:\TRUMAN\Scene_Mesh_ply\\{scene[scene_id]}.ply')
    scene_vertices=scene_mesh.vertices

    print(scene_vertices[:,0].max(),scene_vertices[:,0].min(),scene_vertices[:,1].max(),scene_vertices[:,1].min(),scene_vertices[:,2].max(),scene_vertices[:,2].min())
    x_length=np.round((scene_vertices[:,0].max()-scene_vertices[:,0].min())*10).astype(int)
    if x_length%2!=0:
        x_length+=1
    y_length=np.round((scene_vertices[:,1].max()-scene_vertices[:,1].min())*10)
    if y_length%2!=0:
        y_length+=1
    min_z=scene_vertices[:,2].min()*10

    # 首先将三个关键点转换为标准化的坐标
    transl = np.array([cup_01[scene_id, 0, :3, -1], cup_01[scene_id, 1, :3, -1]])
    obj_translation = transl[:, :2]
    human_transl2obj = np.zeros((2, 3))
    human_transl_dir = fr"D:\Projects\chois_release\scratch\object_A_star\temp_data\data_cup_01_{scene_id:02d}"
    file_name = sorted(os.listdir(human_transl_dir))
    human_transl2obj[0, :] = np.load(fr"D:\Projects\chois_release\scratch\object_A_star\temp_data\data_cup_01_{scene_id:02d}\{file_name[1]}")
    human_transl2obj[1, :] = np.load(fr"D:\Projects\chois_release\scratch\object_A_star\temp_data\data_cup_01_{scene_id:02d}\{file_name[3]}")

    key_points_ori = np.array([
        human_cup_01[scene_id][:2],
        obj_translation[0]+human_transl2obj[0,:2],
        obj_translation[1]+human_transl2obj[1,:2],
    ])
    key_points = np.round(key_points_ori*10+np.array([x_length/2.,y_length/2.])).astype(int)
    # key_points[1] = [20, 87]

    key_z = [0.91, human_transl2obj[0, -1], human_transl2obj[1, -1]]
    
    scene_2d_map=np.zeros((x_length.astype(int),y_length.astype(int)))
    for i in tqdm(range(len(scene_vertices))):
        vert=scene_vertices[i]*10
        x,y,z=vert
        if min_z+2<z<min_z+20:
            try:
                scene_2d_map[np.round(x+x_length/2.).astype(int)][np.round(y+y_length/2.).astype(int)]+=1
            except:
                pass

    
    # 保险机制防止key_points在scene_2d_map上的值大于0
    for i in range(len(key_points)):
        a=random_move(scene_2d_map, key_points[i])
        key_points[i]=a
    

    for index in range(len(key_points)-1):
        node_start = (key_points[index][0], key_points[index][1])
        node_end = (key_points[index+1][0], key_points[index+1][1])

        scene_2d_map_clipped = np.clip(scene_2d_map, a_min=None, a_max=151)
        plt.imshow(scene_2d_map_clipped, cmap='viridis')
        plt.colorbar()
        plt.show()

        # smooth the scene_2d_map
        scene_2d_map_smooth=convolve_2d(scene_2d_map)
        scene_2d_map_smooth_clipped = np.clip(scene_2d_map_smooth, a_min=None, a_max=151)

        # visualize the smoothed scene_2d_map
        plt.imshow(scene_2d_map_smooth_clipped, cmap='viridis')
        plt.colorbar()
        plt.show()

        # use A * to find the path
        path=astar(scene_2d_map_smooth, scene_2d_map, node_start, node_end)

        # visualize the path
        for point in path:
            plt.scatter(point[1], point[0], c='r', marker='o', s=180)
        plt.imshow(scene_2d_map_clipped, cmap='viridis')
        plt.colorbar()
        plt.show()

        # sparse path
        path_sparse=[]
        for i in range(0, len(path), 5):
            path_sparse.append(path[i])
        if path_sparse[0]!=path[0]:
            path_sparse.insert(0, path[0])
        init_len=len(path_sparse)
        while len(path_sparse)%4!=1 or len(path_sparse) == 1:
            path_sparse.insert(0, path[0])
            print(len(path_sparse))
        last_len=len(path_sparse)
        static_len = last_len - init_len

            
        # visualize the sparse path
        for point in path_sparse:
            plt.scatter(point[1], point[0], c='r', marker='o', s=180)
        plt.imshow(scene_2d_map_clipped, cmap='viridis')
        plt.colorbar()
        plt.show()

        # reverse the path
        path_sparse.reverse()
        for i,node in enumerate(path_sparse):
            path_sparse[i]=[(node[0]-x_length/2.)/10.,(node[1]-y_length/2.)/10.,0.91]
        
        if index==0:
            path_sparse[0][2]=key_z[0]
            path_sparse[0][:2]=key_points_ori[0][:2]

            path_sparse[-1-static_len][2]=key_z[1]
            path_sparse[-1][:2]=key_points_ori[1][:2]

            for j in range(1, static_len+1):
                path_sparse[-1-j][:2]=key_points_ori[1][:2]

        elif index==1:
            path_sparse[0][:2]=key_points_ori[1][:2]

            path_sparse[-1-static_len][2]=key_z[2]
            path_sparse[-1][:2]=key_points_ori[2][:2]

            for j in range(1, static_len+1):
                path_sparse[-1-j][:2]=key_points_ori[2][:2]

        else:
            path_sparse[0][2]=key_z[2]
            path_sparse[0][:2]=key_points_ori[2][:2]

            path_sparse[-1][:2]=key_points_ori[3][:2]

            for j in range(1, static_len+1):
                path_sparse[-1-j][:2]=key_points_ori[3][:2]

        # save the path to a file
        path_sparse=np.array(path_sparse)
        print(path_sparse)
        np.save(f'demo/data/key_frames/data_cup_01_{scene_id:02d}/cup_01_{scene_id:02d}_{index:02d}.npy', path_sparse)
    


