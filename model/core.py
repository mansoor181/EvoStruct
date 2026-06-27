import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

import esm as esm_module
import numpy as np
from data.pdb_utils import VOCAB
from model.modules import RelationEGNN


# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation) for ESM-2 transformer layers
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear (frozen)."""

    def __init__(self, base_linear, rank=4, alpha=1.0):
        super().__init__()
        self.base_linear = base_linear
        in_f, out_f = base_linear.in_features, base_linear.out_features
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(in_f, rank, device=device) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=device))
        self.scale = alpha / rank

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out

# ESM-2 model configs: (repr_layer, embedding_dim)
ESM_CONFIGS = {
    'esm2_t6_8M_UR50D': (6, 320),
    'esm2_t12_35M_UR50D': (12, 480),
    'esm2_t30_150M_UR50D': (30, 640),
    'esm2_t33_650M_UR50D': (33, 1280),
    'esm2_t36_3B_UR50D': (36, 2560),
}


class PosEmbedding(nn.Module):

    def __init__(self, num_embeddings):
        super(PosEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = E_idx.unsqueeze(-1) * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


class ProteinFeature(nn.Module):

    def __init__(self, interface_only):
        super().__init__()
        # global nodes and mask nodes
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)

        # segment ids
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 3, 1, 2

        # positional embedding
        self.node_pos_embedding = PosEmbedding(16)
        self.edge_pos_embedding = PosEmbedding(16)
        self.interface_only = interface_only

    def _construct_segment_ids(self, S):
        # construct segment ids. 1/2/3 for antigen/heavy chain/light chain
        glbl_node_mask = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id

        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)

        segment_idx = torch.zeros_like(S)
        segment_idx[glbl_node_mask] = 1.0
        segment_mask = torch.cumsum(segment_idx, dim=0)

        return segment_ids, segment_mask, torch.nonzero(segment_idx)[:, 0]

    def _radial_edges(self, X, src_dst, cutoff):
        dist = X[:, 1][src_dst]  # [Ef, 2, 3], CA position
        dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1) # [Ef]
        src_dst = src_dst[dist <= cutoff]
        src_dst = src_dst.transpose(0, 1)  # [2, Ef]
        return src_dst

    def _knn_edges(self, X, offsets, segment_ids, is_global, top_k=5, eps=1e-6):

        for batch in range(len(offsets)):
            if batch != len(offsets) - 1:
                X_batch = X[offsets[batch]:offsets[batch+1], 1, :]
            else:
                X_batch = X[offsets[batch]:, 1, :]

            dX = torch.unsqueeze(X_batch, 0) - torch.unsqueeze(X_batch, 1)
            D = torch.sqrt(torch.sum(dX**2, 2) + eps)
            _, E_idx = torch.topk(D, top_k, dim=-1, largest=False)

            if batch == 0:
                row = torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1)
                col = E_idx.view(-1)
            else:
                row = torch.cat([row, torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1) + offsets[batch]], dim=0)
                col = torch.cat([col, E_idx.view(-1) + offsets[batch]], dim=0)

        row_seg, col_seg = segment_ids[row], segment_ids[col]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_edges_knn = torch.stack([row[select_edges], col[select_edges]])

        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_edges_knn = torch.stack([row[select_edges], col[select_edges]])

        return ctx_edges_knn, inter_edges_knn

    def get_node_pos(self, X, segment_mask, segment_idx):
        pos = torch.arange(X.shape[0], device=X.device) - segment_idx[segment_mask-1]
        pos_node_feats = self.node_pos_embedding(pos.view(1, X.shape[0], 1))[0, :, 0, :]  # [1, N, 1, 16] -> [N, 16]

        return pos_node_feats

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., 16
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF

    def get_node_dist(self, X, eps=1e-6):
        d_NC = torch.sqrt(torch.sum((X[:, 0, :] - X[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X[:, 2, :] - X[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X[:, 3, :] - X[:, 1, :])**2, dim=1) + eps)

        d_NC_RBF = self._rbf(d_NC)
        d_CC_RBF = self._rbf(d_CC)
        d_OC_RBF = self._rbf(d_OC)

        dis_node_feats = torch.cat((d_NC_RBF, d_CC_RBF, d_OC_RBF), 1)

        return dis_node_feats

    def get_node_angle(self, X, segment_idx, segment_ids, eps=1e-6):

        # First 3 coordinates are N, CA, C
        X = X[:, :3,:].reshape(1, 3*X.shape[0], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]

        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)

        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)

        # psi, omega, phi
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Dihedral_Angle_features  = torch.cat((torch.cos(D), torch.sin(D)), 2)


        cosD = (u_2*u_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)

        # beta, alpha, gamma
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

        angle_node_feats = torch.cat((Dihedral_Angle_features, Angle_features), 2)[0]

        for i in segment_idx:
            if i == 0:
                angle_node_feats[i:i+2] = 0
            else:
                angle_node_feats[i-1:i+2] = 0

        if self.interface_only == 0:
            angle_node_feats[segment_ids == self.ag_seg_id] = 0

        return angle_node_feats

    def get_node_direct(self, Xs, segment_idx, segment_ids):
        X = Xs[:, 1,:].reshape(1, Xs.shape[0], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:] # # CA-N, C-CA, N-C, CA-N ...
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]

        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        o_1 = F.normalize(u_2 - u_1, dim=-1)

        # Build relative orientations
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)
        O = O.view(list(O.shape[:2]) + [3,3])

        for i in segment_idx:
            if i == 0:
                O[:, i:i+2, :, :] = 0
            else:
                O[:, i-2:i+2, :, :] = 0

        if self.interface_only == 0:
            O[:, segment_ids == self.ag_seg_id, :, :] = 0

        # Rotate into local reference frames
        d_NC = (Xs[:, 0, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O, d_NC).squeeze(-1), dim=-1)

        d_CC = (Xs[:, 2, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O, d_CC).squeeze(-1), dim=-1)

        d_OC = (Xs[:, 3, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O, d_OC).squeeze(-1), dim=-1)

        direct_node_feats = torch.cat((d_NC, d_CC, d_OC), 2)[0]

        return direct_node_feats, O

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
        """

        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes

        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def get_edge_pos(self, edge_index):
        pos = (edge_index[0:1, :] - edge_index[1:2, :]).float().unsqueeze(-1)
        pos_edge_feats = self.edge_pos_embedding(pos)[0, :, 0, :]  # [1, E, 1, 16] -> [E, 16]

        return pos_edge_feats

    def get_edge_dist(self, X, edge_index, eps=1e-6):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]

        d_NC = torch.sqrt(torch.sum((X_row[:, 0, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CAC = torch.sqrt(torch.sum((X_row[:, 1, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X_row[:, 2, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X_row[:, 3, :] - X_col[:, 1, :])**2, dim=1) + eps)

        d_NC_RBF = self._rbf(d_NC)
        d_CAC_RBF = self._rbf(d_CAC)
        d_CC_RBF = self._rbf(d_CC)
        d_OC_RBF = self._rbf(d_OC)

        dis_edge_feats = torch.cat((d_NC_RBF, d_CAC_RBF, d_CC_RBF, d_OC_RBF), 1)

        return dis_edge_feats

    def get_edge_angle(self, O, edge_index):

        O_row, O_col = O[:, edge_index[0, :], :, :].unsqueeze(2), O[:, edge_index[1, :], :, :].unsqueeze(2)
        R = torch.matmul(O_row.transpose(-1,-2), O_col)
        angle_edge_feats = self._quaternions(R)[0, :, 0, :]

        return angle_edge_feats

    def get_edge_direct(self, X, O, edge_index):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]
        _, O_col = O[:, edge_index[0, :], :, :], O[:, edge_index[1, :], :, :]

        # Rotate into local reference frames
        d_NC = (X_row[:, 0, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O_col, d_NC).squeeze(-1), dim=-1)

        d_CAC = (X_row[:, 1, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CAC = F.normalize(torch.matmul(O_col, d_CAC).squeeze(-1), dim=-1)

        d_CC = (X_row[:, 2, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O_col, d_CC).squeeze(-1), dim=-1)

        d_OC = (X_row[:, 3, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O_col, d_OC).squeeze(-1), dim=-1)

        direct_edge_feats = torch.cat((d_NC, d_CAC, d_CC, d_OC), 2)[0]

        return direct_edge_feats

    def edge_masking(self, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, edge_type):

        if edge_type == 1 or edge_type == 2 or edge_type == 6 or edge_type == 7:
            pos_edge_feats *= 0

        return pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]

        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # prepare inputs
        segment_ids, segment_mask, segment_idx = self._construct_segment_ids(S)

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)

        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index

        # not global edges
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        # all possible ctx edges: same seg, not global
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]

        ctx_edges_rball = self._radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=8.0)
        ctx_edges_knn, inter_edges_knn = self._knn_edges(X, offsets, segment_ids, is_global, top_k=8)

        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges)
        ctx_edges_seq_d1 = torch.stack([row[select_edges_seq], col[select_edges_seq]])

        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges)
        ctx_edges_seq_d2 = torch.stack([row[select_edges_seq], col[select_edges_seq]])

        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges_rball = self._radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=12.0)

        # edges between global and normal nodes
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        # edges between global and global nodes
        select_edges = torch.logical_and(row_global, col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        # construct node features
        pos_node_feats = self.get_node_pos(X, segment_mask, segment_idx)
        dis_node_feats = self.get_node_dist(X)
        angle_node_feats = self.get_node_angle(X, segment_idx.tolist(), segment_ids)
        direct_node_feats, O = self.get_node_direct(X, segment_idx.tolist(), segment_ids)
        node_feats = torch.cat((pos_node_feats, dis_node_feats, angle_node_feats, direct_node_feats), 1)

        # construct edge features
        edges_list = [ctx_edges_rball, global_normal, global_global, ctx_edges_seq_d1, ctx_edges_knn, ctx_edges_seq_d2, inter_edges_rball, inter_edges_knn]
        edge_class_type = torch.eye(len(edges_list), dtype=torch.float, device=X.device)
        edge_feats_list = []

        for i in range(len(edges_list)):
            type_edge_feats = edge_class_type[torch.ones(edges_list[i].shape[1]).long() * i]
            pos_edge_feats = self.get_edge_pos(edges_list[i])
            dis_edge_feats = self.get_edge_dist(X, edges_list[i])
            angle_edge_feats = self.get_edge_angle(O, edges_list[i])
            direct_edge_feats = self.get_edge_direct(X, O, edges_list[i])
            pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats = self.edge_masking(pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, i)
            edge_feats = torch.cat((type_edge_feats, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats), 1)
            edge_feats_list.append(edge_feats)

        return edges_list, edge_feats_list, node_feats, segment_idx, segment_ids

    def forward(self, X, S, offsets):
        batch_id = torch.zeros_like(S)
        batch_id[offsets[1:-1]] = 1
        batch_id = torch.cumsum(batch_id, dim=0)

        return self.construct_edges(X, S, batch_id)


class MiniStructuralAdapter(nn.Module):
    """Mini adapter: 1 block, reduced dimension (640).

    - Only 1 cross-attention block
    - Works at adapter_dim (640) instead of esm_dim (1280)
    - Projects ESM down to adapter_dim, processes, projects back up

    This reduces adapter params from ~40M to ~8M while keeping the
    cross-attention mechanism that injects structural context into PLM states.
    """

    def __init__(self, esm_dim, adapter_dim, gnn_dim, n_heads, dropout):
        super().__init__()
        self.esm_down = nn.Linear(esm_dim, adapter_dim)
        self.gnn_to_adapter = nn.Linear(gnn_dim, adapter_dim)
        self.norm_q = nn.LayerNorm(adapter_dim)
        self.norm_kv = nn.LayerNorm(adapter_dim)
        self.attn = nn.MultiheadAttention(
            adapter_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(adapter_dim)
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )
        self.esm_up = nn.Linear(adapter_dim, esm_dim)

    def forward(self, esm_cdr, gnn_ctx):
        """esm_cdr: (L_cdr, esm_dim) | gnn_ctx: (L_ctx, gnn_dim)"""
        # Project down to adapter dimension
        q = self.esm_down(esm_cdr).unsqueeze(0)   # (1, L_cdr, adapter_dim)
        kv = self.gnn_to_adapter(gnn_ctx).unsqueeze(0)  # (1, L_ctx, adapter_dim)

        # Single cross-attention block
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)
        attn_out, _ = self.attn(q_in, kv_in, kv_in)
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))

        # Project back up to esm_dim
        return self.esm_up(q.squeeze(0))  # (L_cdr, esm_dim)


class DockDesigner(nn.Module):
    """Mini structural adapter (1 block, dim 640) + shadow paratope.

    The seq_head still operates at esm_dim to preserve PLM vocabulary output.
    """

    def __init__(self, embed_size, hidden_size, n_channel, n_layers, dropout, cdr_type, args):
        super().__init__()
        self.cdr_type = cdr_type
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = getattr(args, 'gamma', 0.5)
        self.delta = getattr(args, 'delta', 0.3)
        self.dock_cutoff = getattr(args, 'dock_cutoff', 8.0)
        self.num_attn_heads = getattr(args, 'num_attn_heads', 4)
        self.label_smoothing = getattr(args, 'label_smoothing', 0.0)
        # v30h: mini adapter params
        self.adapter_n_blocks = getattr(args, 'adapter_n_blocks', 1)  # Only 1 block
        self.adapter_dim = getattr(args, 'adapter_dim', 640)  # Reduced dimension
        self.adapter_n_heads = getattr(args, 'adapter_n_heads', 8)
        self.adapter_max_ag_ctx = getattr(args, 'adapter_max_ag_ctx', 128)
        node_feats_mode = args.node_feats_mode
        edge_feats_mode = args.edge_feats_mode
        self.interface_only = args.interface_only

        node_feats_dim = int(node_feats_mode[0]) * 16 + int(node_feats_mode[1]) * 48 + int(node_feats_mode[2]) * 12 + int(node_feats_mode[3]) * 9
        edge_feats_dim = int(edge_feats_mode[0]) * 16 + int(edge_feats_mode[1]) * 64 + int(edge_feats_mode[2]) * 4 + int(edge_feats_mode[3]) * 12 + 8

        self.num_aa_type = len(VOCAB)
        self.mask_token_id = VOCAB.get_unk_idx()
        self.projection_head_cdr = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.SiLU(), nn.Linear(hidden_size//2, hidden_size//4))
        self.projection_head_ant = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.SiLU(), nn.Linear(hidden_size//2, hidden_size//4))

        self.aa_embedding = nn.Embedding(self.num_aa_type, embed_size)
        self.gnn = RelationEGNN(embed_size, hidden_size, self.num_aa_type, n_channel, n_layers=n_layers, dropout=dropout, node_feats_dim=node_feats_dim, edge_feats_dim=edge_feats_dim)

        # ESM-2 language model (frozen by default, top layers unfreezable)
        esm_model_name = getattr(args, 'esm_model_name', 'esm2_t33_650M_UR50D')
        self.esm_model_name = esm_model_name
        self.esm_repr_layer, self.esm_dim = ESM_CONFIGS[esm_model_name]
        esm_loader = getattr(esm_module.pretrained, esm_model_name)
        self.esm_model, self.esm_alphabet = esm_loader()
        self.esm_model.eval()
        for p in self.esm_model.parameters():
            p.requires_grad = False
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

        # v30h: Mini structural adapter (1 block, reduced dim)
        self.adapter = MiniStructuralAdapter(
            esm_dim=self.esm_dim,
            adapter_dim=self.adapter_dim,
            gnn_dim=hidden_size,
            n_heads=self.adapter_n_heads,
            dropout=dropout,
        )

        # Sequence head operates at esm_dim (output from adapter)
        self.seq_head = nn.Sequential(
            nn.LayerNorm(self.esm_dim),
            nn.Linear(self.esm_dim, self.esm_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.esm_dim // 2, self.num_aa_type),
        )

        self.protein_feature = ProteinFeature(args.interface_only)

    def set_esm_unfreeze_top(self, n_top):
        """Unfreeze the top `n_top` transformer layers of ESM-2 (phase 2)."""
        total = len(self.esm_model.layers)
        n_top = min(n_top, total)
        for i, layer in enumerate(self.esm_model.layers):
            trainable = i >= (total - n_top)
            for p in layer.parameters():
                p.requires_grad = trainable
        if hasattr(self.esm_model, "emb_layer_norm_after") and n_top > 0:
            for p in self.esm_model.emb_layer_norm_after.parameters():
                p.requires_grad = True
        return sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)

    def inject_lora(self, n_top, rank=4, alpha=1.0):
        """Inject LoRA adapters into top `n_top` ESM-2 layers (q, k, v, out projections).
        Keeps all ESM weights frozen; only LoRA params are trainable."""
        total = len(self.esm_model.layers)
        n_top = min(n_top, total)
        n_lora_params = 0
        for i in range(total - n_top, total):
            layer = self.esm_model.layers[i]
            attn = layer.self_attn
            for attr in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                base = getattr(attn, attr)
                lora = LoRALinear(base, rank=rank, alpha=alpha)
                setattr(attn, attr, lora)
                n_lora_params += lora.lora_A.numel() + lora.lora_B.numel()
        # Ensure base ESM weights stay frozen, only LoRA params trainable
        for p in self.esm_model.parameters():
            p.requires_grad = False
        for m in self.esm_model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True
        return n_lora_params

    def set_neftune(self, alpha=5.0):
        """Enable NEFTune: adds uniform noise to ESM embeddings during training."""
        self._neftune_alpha = alpha

    def _neftune_hook(self, embeddings):
        """Add NEFTune noise to embeddings during training."""
        if self.training and hasattr(self, '_neftune_alpha') and self._neftune_alpha > 0:
            dims = embeddings.shape
            mag = self._neftune_alpha / (dims[-2] * dims[-1]) ** 0.5
            embeddings = embeddings + torch.zeros_like(embeddings).uniform_(-mag, mag)
        return embeddings

    def state_dict(self, *args, **kwargs):
        """Save only ESM-2 params that are trainable (top layers in phase 2+),
        and all other (non-esm) params. Frozen ESM-2 weights come from the
        pretrained checkpoint when loading.
        """
        full_state = super().state_dict(*args, **kwargs)
        trainable_esm_keys = set()
        for name, p in self.esm_model.named_parameters():
            if p.requires_grad:
                trainable_esm_keys.add(f"esm_model.{name}")
        return {k: v for k, v in full_state.items()
                if (not k.startswith('esm_model.')) or (k in trainable_esm_keys)}

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, ignoring missing ESM-2 keys and shape mismatches
        (only trainable top-layer ESM params, if any, are expected in the checkpoint)."""
        model_state = self.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        return super().load_state_dict(filtered, strict=False)

    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction='none',
                               label_smoothing=self.label_smoothing)

    def coord_loss(self, _input, target):
        return F.smooth_l1_loss(_input, target, reduction='sum')

    def init_mask(self, X, S, cdr_range):
        X, S, cmask = X.clone(), S.clone(), torch.zeros_like(X, device=X.device)
        n_channel, n_dim = X.shape[1:]
        for start, end in cdr_range:
            S[start:end + 1] = self.mask_token_id
            l_coord, r_coord = X[start - 1], X[end + 1]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = 1
        return X, S, cmask

    def _get_esm_cdr_embeddings(self, S, segment_ids, cdr_range, offsets):
        """Get ESM-2 embeddings for CDR positions using HC context."""
        device = S.device
        hc_seg_id = self.protein_feature.hc_seg_id
        is_global = sequential_or(
            S == self.protein_feature.boa_idx,
            S == self.protein_feature.boh_idx,
            S == self.protein_feature.bol_idx)

        esm_cdr_parts = []

        for i, (cdr_start, cdr_end) in enumerate(cdr_range):
            c_start = offsets[i].item()
            c_end = offsets[i + 1].item() if i + 1 < len(offsets) else len(S)
            cdr_len = cdr_end.item() - cdr_start.item() + 1

            # HC non-global positions in this complex
            c_seg = segment_ids[c_start:c_end]
            c_global = is_global[c_start:c_end]
            hc_mask = (c_seg == hc_seg_id) & ~c_global
            hc_local_idx = torch.where(hc_mask)[0]
            hc_global_idx = hc_local_idx + c_start

            if len(hc_global_idx) == 0:
                esm_cdr_parts.append(
                    torch.zeros(cdr_len, self.esm_dim, device=device))
                continue

            # Map CDR global positions to HC-local positions
            g2hc = {g.item(): j for j, g in enumerate(hc_global_idx)}
            cdr_pos_in_hc = [g2hc[p] for p in range(
                cdr_start.item(), cdr_end.item() + 1) if p in g2hc]

            if not cdr_pos_in_hc:
                esm_cdr_parts.append(
                    torch.zeros(cdr_len, self.esm_dim, device=device))
                continue

            # Build HC sequence string; masked CDR positions get placeholder
            hc_tokens = S[hc_global_idx]
            hc_chars = []
            mask_pos_in_hc = []
            for j, t in enumerate(hc_tokens):
                if t.item() == self.mask_token_id:
                    mask_pos_in_hc.append(j)
                    hc_chars.append('A')  # placeholder, will be ESM-masked
                else:
                    sym = VOCAB.idx_to_symbol(t.item())
                    hc_chars.append(
                        sym if (len(sym) == 1 and sym.isalpha()) else 'X')

            # Tokenize for ESM-2 and apply masks
            _, _, esm_tok = self.esm_batch_converter(
                [("hc", ''.join(hc_chars))])
            esm_tok = esm_tok.to(device)
            for p in mask_pos_in_hc:
                esm_tok[0, p + 1] = self.esm_alphabet.mask_idx  # +1 for <cls>

            # Run ESM-2 (frozen base, or with LoRA adapters)
            esm_out = self.esm_model(
                esm_tok, repr_layers=[self.esm_repr_layer])
            esm_repr = esm_out["representations"][self.esm_repr_layer]
            esm_repr = esm_repr[0, 1:len(hc_chars) + 1, :]  # (L_hc, esm_dim)
            esm_repr = self._neftune_hook(esm_repr)

            esm_cdr_parts.append(esm_repr[cdr_pos_in_hc])  # (L_cdr, esm_dim)

        return torch.cat(esm_cdr_parts, dim=0)  # (total_cdr, esm_dim)

    def _contrastive_pairing_loss(self, aa_embd, S, cdr_range, segment_idx):
        """InfoNCE loss between CDR and antigen embeddings (from RAAD)."""
        aa_embd_cdr = self.projection_head_cdr(aa_embd)
        aa_embd_ant = self.projection_head_ant(aa_embd)

        cdr_logits = []
        for start, end in cdr_range:
            cdr_logits.append(torch.mean(aa_embd_cdr[start:end+1], dim=0, keepdim=True))
        cdr_logits = torch.cat(cdr_logits, dim=0)

        ant_logits = []
        segment_list = segment_idx.tolist()
        for i, index in enumerate(segment_list):
            if S[index] == VOCAB.symbol_to_idx(VOCAB.BOA):
                if index == segment_list[-1]:
                    ant_logits.append(torch.mean(aa_embd_ant[index:], dim=0, keepdim=True))
                else:
                    ant_logits.append(torch.mean(aa_embd_ant[index:segment_list[i+1]], dim=0, keepdim=True))

        if len(ant_logits) == 0 or len(ant_logits) != len(cdr_logits):
            return torch.tensor(0.0, device=aa_embd.device)

        ant_logits = torch.cat(ant_logits, dim=0)

        norm1, norm2 = cdr_logits.norm(dim=1), ant_logits.norm(dim=1)
        mat_norm = torch.einsum('i,j->ij', norm1, norm2)
        mat_sim = torch.exp(torch.einsum('ik,jk,ij->ij', cdr_logits, ant_logits, 1/mat_norm.clamp(min=1e-8)) / 1.0)

        b = cdr_logits.size(0)
        diag = mat_sim[range(b), range(b)]
        p_loss = -torch.log(diag / (mat_sim.sum(dim=1) - diag).clamp(min=1e-8)).mean()
        return p_loss

    def _dock_loss(self, Z, true_X, cdr_range, S, segment_ids):
        """Docking proximity loss: predicted CDR should be near epitope."""
        device = Z.device
        is_global = sequential_or(S == self.protein_feature.boa_idx,
                                  S == self.protein_feature.boh_idx,
                                  S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        ag_ca = true_X[ag_mask, 1, :]  # (N_ag, 3) -- antigen CA unchanged

        if ag_ca.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        losses = []
        for start, end in cdr_range:
            true_cdr_ca = true_X[start:end + 1, 1, :]  # (L_cdr, 3)
            pred_cdr_ca = Z[start:end + 1, 1, :]        # (L_cdr, 3)

            # Epitope: AG residues within cutoff of any true CDR CA
            true_dists = torch.cdist(ag_ca, true_cdr_ca)  # (N_ag, L_cdr)
            epi_mask = true_dists.min(dim=1).values < self.dock_cutoff
            if not epi_mask.any():
                continue

            epi_ca = ag_ca[epi_mask]  # (N_epi, 3)
            # Min distance from each predicted CDR CA to nearest epitope CA
            pred_dists = torch.cdist(pred_cdr_ca, epi_ca)  # (L_cdr, N_epi)
            min_dists = pred_dists.min(dim=1).values       # (L_cdr,)
            losses.append(min_dists.mean())

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _adapter_logits(self, aa_embd, true_X, S, segment_ids, offsets,
                        cdr_range, esm_cdr_embd):
        """Structural-adapter-based logits."""
        is_global = sequential_or(
            S == self.protein_feature.boa_idx,
            S == self.protein_feature.boh_idx,
            S == self.protein_feature.bol_idx)
        ag_seg_id = self.protein_feature.ag_seg_id

        all_logits = []
        cdr_offset = 0
        for i, (start, end) in enumerate(cdr_range):
            cdr_h = aa_embd[start:end + 1]            # (L_cdr, gnn_dim)
            cdr_len = end - start + 1

            c_start = offsets[i].item()
            c_end = offsets[i + 1].item() if (i + 1) < len(offsets) else aa_embd.shape[0]
            c_seg = segment_ids[c_start:c_end]
            c_global = is_global[c_start:c_end]
            ag_local_idx = torch.where((c_seg == ag_seg_id) & ~c_global)[0]

            if ag_local_idx.numel() > 0:
                ag_global_idx = ag_local_idx + c_start
                ag_ca = true_X[ag_global_idx, 1, :]           # (N_ag, 3)
                cdr_ca = true_X[start:end + 1, 1, :]          # (L_cdr, 3)
                d = torch.cdist(ag_ca, cdr_ca).min(dim=1).values  # (N_ag,)
                k = min(self.adapter_max_ag_ctx, ag_global_idx.numel())
                top = torch.topk(-d, k).indices
                ag_h = aa_embd[ag_global_idx[top]]            # (k, gnn_dim)
                gnn_ctx = torch.cat([cdr_h, ag_h], dim=0)
            else:
                gnn_ctx = cdr_h

            esm_q = esm_cdr_embd[cdr_offset:cdr_offset + cdr_len]  # (L_cdr, esm_dim)
            refined = self.adapter(esm_q, gnn_ctx)                  # (L_cdr, esm_dim)
            logits = self.seq_head(refined)                          # (L_cdr, num_aa)
            all_logits.append(logits)
            cdr_offset += cdr_len

        return torch.cat(all_logits, dim=0)

    def _shadow_paratope_loss(self, Z, true_X, cdr_range, S, segment_ids):
        """dyMEAN shadow paratope loss."""
        device = Z.device
        is_global = sequential_or(S == self.protein_feature.boa_idx,
                                  S == self.protein_feature.boh_idx,
                                  S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        ag_ca = true_X[ag_mask, 1, :]
        if ag_ca.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        losses = []
        for start, end in cdr_range:
            true_cdr_ca = true_X[start:end + 1, 1, :]
            pred_cdr_ca = Z[start:end + 1, 1, :]

            true_d = torch.cdist(ag_ca, true_cdr_ca)
            epi_mask = true_d.min(dim=1).values < self.dock_cutoff
            if not epi_mask.any():
                continue
            epi_ca = ag_ca[epi_mask]

            true_dm = torch.cdist(true_cdr_ca, epi_ca)  # (L_cdr, N_epi)
            pred_dm = torch.cdist(pred_cdr_ca, epi_ca)  # (L_cdr, N_epi)
            losses.append((pred_dm - true_dm).abs().mean())

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def forward(self, X, S, L, offsets):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        true_X, true_S = X.clone(), S.clone()
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()
        aa_cnt = mask.sum()

        H_0 = self.aa_embedding(S)

        with torch.no_grad():
            edges_list, edge_feats_list, node_feats, segment_idx, segment_ids = self.protein_feature(X, S, offsets)
        H, Z, aa_embd = self.gnn(H_0, X, edges_list, edge_feats_list, node_feats, segment_ids, self.interface_only)

        esm_cdr_embd = self._get_esm_cdr_embeddings(
            S, segment_ids, cdr_range, offsets)

        cdr_logits = self._adapter_logits(
            aa_embd, true_X, S, segment_ids, offsets, cdr_range, esm_cdr_embd)
        seq_loss = torch.sum(self.seq_loss(cdr_logits, true_S[mask])) / aa_cnt
        coord_loss = self.coord_loss(Z[mask], true_X[mask]) / aa_cnt

        if len(cdr_range) > 1 and (S == VOCAB.symbol_to_idx(VOCAB.BOA)).sum() == len(cdr_range):
            pairing_loss = self._contrastive_pairing_loss(aa_embd, S, cdr_range, segment_idx)
        else:
            pairing_loss = torch.tensor(0.0, device=X.device)

        dock_loss = self._dock_loss(Z, true_X, cdr_range, S, segment_ids)
        shadow_loss = self._shadow_paratope_loss(Z, true_X, cdr_range, S, segment_ids)

        loss = (seq_loss
                + self.alpha * coord_loss
                + self.beta * pairing_loss
                + self.gamma * dock_loss
                + self.delta * shadow_loss)

        return loss, seq_loss, coord_loss, pairing_loss, dock_loss, shadow_loss

    def generate(self, X, S, L, offsets, greedy=True):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        true_X, true_S = X.clone(), S.clone()
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)

        with torch.no_grad():
            edges_list, edge_feats_list, node_feats, segment_idx, segment_ids = self.protein_feature(X, S, offsets)
        H, Z, aa_embd = self.gnn(H_0, X, edges_list, edge_feats_list, node_feats, segment_ids, self.interface_only)

        X = X.clone()
        X[mask] = Z[mask]

        esm_cdr_embd = self._get_esm_cdr_embeddings(
            S, segment_ids, cdr_range, offsets)

        logits = self._adapter_logits(
            aa_embd, X, S, segment_ids, offsets, cdr_range, esm_cdr_embd)
        logits = logits.masked_fill(smask, float('-inf'))

        if greedy:
            S[mask] = torch.argmax(logits, dim=-1)
        else:
            prob = F.softmax(logits, dim=-1)
            S[mask] = torch.multinomial(prob, num_samples=1).squeeze()
        snll_all = self.seq_loss(logits, S[mask])

        return snll_all, S, X, true_X, cdr_range, logits

    def infer(self, batch, device, greedy=True):
        X, S, L, offsets = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device)
        snll_all, pred_S, pred_X, true_X, cdr_range, logits = self.generate(X, S, L, offsets, greedy=greedy)

        pred_S, cdr_range = pred_S.tolist(), cdr_range.tolist()
        pred_X, true_X = pred_X.detach().cpu().numpy(), true_X.detach().cpu().numpy()

        seq, x, true_x, logits_list = [], [], [], []
        logit_offset = 0
        for start, end in cdr_range:
            end = end + 1
            length = end - start
            seq.append(''.join([VOCAB.idx_to_symbol(pred_S[i]) for i in range(start, end)]))
            x.append(pred_X[start:end])
            true_x.append(true_X[start:end])
            logits_list.append(logits[logit_offset:logit_offset + length].detach().cpu())
            logit_offset += length

        ppl = [0 for _ in range(len(cdr_range))]
        lens = [0 for _ in ppl]
        offset = 0
        for i, (start, end) in enumerate(cdr_range):
            length = end - start + 1
            for t in range(length):
                ppl[i] += snll_all[t + offset]
            offset += length
            lens[i] = length

        ppl = [p / n for p, n in zip(ppl, lens)]
        ppl = torch.exp(torch.tensor(ppl, device=device)).tolist()

        return ppl, seq, x, true_x, logits_list
