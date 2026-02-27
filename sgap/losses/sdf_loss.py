from pytorch3d.structures import Meshes
import torch
from pytorch3d.ops import knn_points

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

if __name__ == '__main__':
    obj_out = obj_verts[i]

    # 构建mesh
    mesh_l = Meshes(verts=mano_l_verts[i], faces=self.seal_face_l.repeat(mano_l_verts[i].shape[0],1,1).cuda())
    mesh_r = Meshes(verts=mano_r_verts[i], faces=self.seal_face_r.repeat(mano_r_verts[i].shape[0],1,1).cuda())
    # 得到mesh的表面法线
    hand_normal_l = mesh_l.verts_normals_packed().view(-1, 778, 3)
    hand_normal_r = mesh_r.verts_normals_packed().view(-1, 778, 3)
    # 得到手和物体的NN距离和索引
    nn_dist_l, nn_idx_l = get_NN(obj_out, mano_l_verts[i])
    nn_dist_r, nn_idx_r = get_NN(obj_out, mano_r_verts[i])

    # 得到手和物体的内部点
    interior_l = get_interior(hand_normal_l, mano_l_verts[i], obj_out, nn_idx_l).type(torch.bool) * padding_mask[i, 0, 1:][:,None]
    interior_r = get_interior(hand_normal_r, mano_r_verts[i], obj_out, nn_idx_r).type(torch.bool) * padding_mask[i, 0, 1:][:,None]

    # 计算内部点的距离
    penetr_dist_l =  nn_dist_l[interior_l]
    penetr_dist_r =  nn_dist_r[interior_r]

    if penetr_dist_l.nelement() != 0:
        pene_l_loss = pene_l_loss + penetr_dist_l.mean()
    if penetr_dist_r.nelement() !=0:
        pene_r_loss = pene_r_loss + penetr_dist_r.mean()