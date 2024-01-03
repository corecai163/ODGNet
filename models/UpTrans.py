#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Pingping Cai

import torch
import torch.nn as nn
import math
from extensions.chamfer_dist import ChamferDistanceL2,ChamferDistanceL1

from models.pointnet import PointNet_SA_Module_KNN, MLP_Res, UpLayer, fps_subsample, Transformer

from .build import MODELS


class UNet(nn.Module):
    def __init__(self, dim_feat=384, num_seeds = 1024, seed_fea=128, num_dicts=256):
        '''
        Extract information from partial point cloud
        '''
        super(UNet, self).__init__()
        self.num_seed = num_seeds
        self.sa_module_1 = PointNet_SA_Module_KNN(num_seeds//2, 16, 3, [32, seed_fea], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(seed_fea, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(num_seeds//8, 16, seed_fea, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        #self.sa_module_3 = PointNet_SA_Module_KNN(32, 16, 256, [256, 384], group_all=False, if_bn=False, if_idx=True)
        #self.transformer_3 = Transformer(384, dim=64)
        self.sa_module_4 = PointNet_SA_Module_KNN(None, None, 256, [384, dim_feat], group_all=True, if_bn=False)
        
        self.ps_0 = nn.ConvTranspose1d(dim_feat, 256, num_seeds//16, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 256, hidden_dim=256, out_dim=256)

        self.ps_1 = nn.ConvTranspose1d(256, 256, 2,2, bias=True)
        self.ps_2 = nn.ConvTranspose1d(256, 128, 2,2, bias=True)
        

        #self.refine_1 = nn.ModuleList([FeaRefine(dim=256,hidd_dim=64,num_dicts=num_dicts) for i in range(2)])
        #self.refine_2 = nn.ModuleList([FeaRefine(dim=128,hidd_dim=64,num_dicts=num_dicts//2) for i in range(2)])

        self.refine_1 = FeaRefine(dim=256,hidd_dim=64,num_dicts=num_dicts) 
        self.refine_2 = FeaRefine(dim=128,hidd_dim=64,num_dicts=num_dicts//2)

        #self.ps_3 = nn.ConvTranspose1d(64, 32, 2, bias=True)
        #self.mlp_1 = MLP_Res(in_dim=dim_feat + 256, hidden_dim=256, out_dim=256)
        #self.mlp_2 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=64)
        self.mlp_3 = MLP_Res(in_dim=seed_fea, hidden_dim=32, out_dim=3)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        
        l0_xyz = point_cloud
        l0_fea = point_cloud

        ## Encoder        
        l1_xyz, l1_fea, idx1 = self.sa_module_1(l0_xyz, l0_fea)  # (B, 3, 512), (B, 128, 512)
        l1_fea = self.transformer_1(l1_fea, l1_xyz)
        l2_xyz, l2_fea, idx2 = self.sa_module_2(l1_xyz, l1_fea)  # (B, 3, 128), (B, 256, 128)
        l2_fea = self.transformer_2(l2_fea, l2_xyz)
        #l3_xyz, l3_points, idx3 = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 32), (B, 384, 32)
        #l3_points = self.transformer_3(l3_points, l3_xyz)
        l4_xyz, l4_fea = self.sa_module_4(l2_xyz, l2_fea)  # (B, 3, 1), (B, out_dim, 1)
        
        ## Decoder
        #seed generate 0
        u0_fea = self.ps_0(l4_fea)  # (b, 256, 128)
        u0_fea = self.mlp_1(torch.cat([u0_fea, l4_fea.repeat((1, 1, u0_fea.size(2)))], 1))

        #upconv1
        u1_fea = self.ps_1(u0_fea)
        #cons1 = []
        #for i, blk in enumerate(self.refine_1):
        u1_fea,cons1 = self.refine_1(u1_fea)

        #u1_fea = u1_fea+self.mlp_1(torch.cat([u1_fea, l4_fea.repeat((1, 1, u1_fea.size(2)))], 1)) # (b, 256, 128)
        #u1_xyz = self.mlp_1(u1_fea)
        
        # skip concat
        u1_fea = torch.cat([l2_fea,u1_fea],dim=2)  # (b, 256, 256)
        #u1_xyz = torch.cat([l2_xyz,u1_xyz],dim=2)
        
        #upconv 2
        u2_fea = self.ps_2(u1_fea)  # (b, 64, 512)
        #cons2 = []
        #for i, blk in enumerate(self.refine_2):
        u2_fea,cons2 = self.refine_2(u2_fea)
        #cons2.append(cons)
        #u2_fea = self.mlp_2(torch.cat([u2_fea, l4_fea.repeat((1, 1, u2_fea.size(2)))], 1)) # (b, 64, 512)
        #u2_xyz = self.mlp_2(u2_fea)
        
        # skip concat
        #u2_fea = torch.cat([l1_fea,u2_fea],dim=2)  
        u2_xyz = self.mlp_3(u2_fea)
        
        u2_xyz = torch.cat([l1_xyz,u2_xyz],dim=2) # (b, 3, 1024)
        u2_fea = torch.cat([l1_fea,u2_fea],dim=2) # (b, 64, 1024)

        #upconv 3
        #u3_fea = self.ps_3(u2_fea) # (b, 32, 2048)
        #u3_xyz = self.mlp_3(u3_fea)
        #print(u2_fea.size())
        return l4_fea, u2_xyz, u2_fea, cons1, cons2

class FeaRefine(nn.Module):
    def __init__(self, dim=512, hidd_dim=64, num_dicts=128):
        super(FeaRefine, self).__init__()
        self.num_dicts = num_dicts
        self.dict_embed = nn.Embedding(dim, num_dicts)
        #self.merge = nn.Conv1d(dim+dim_feat, dim, 1)
        self.query = nn.Conv1d(dim, hidd_dim, 1)
        self.key = nn.Conv1d(dim, hidd_dim, 1)
        #self.linear = nn.Conv1d(dim+dim, dim, 1)
    
    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        B,C,N = feat.size()
        #print(feat.size())
        dict_fea = self.dict_embed.weight # (1, dim, 128)
        #shape_code = coarse_shapecode.repeat(1,1,128)
        feature_memory = torch.nn.functional.normalize(dict_fea,dim=0)
        cons = feature_memory.transpose(0,1) @ feature_memory
        #Iden = torch.eye(self.num_dict,device='cuda')

        dict_fea = dict_fea.unsqueeze(0).repeat(B,1,1)
        
        q = self.query(feat)  # (b, 64, N)
        #query_embed.unsqueeze(0) # (1, 512, num_dicts)
        value = dict_fea #self.merge(torch.cat([shape_code,dict_fea],dim=1)) # (1, dim, 128)
        k = self.key(value) # (1, 64, 128)
        
        d_k = k.size(1)
        # print(agg.size())
        # qk_rel = q - k
        # w = self.weight_mlp(qk_rel)
        # w = torch.softmax(w, -1)
        
        # Attention
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k) # (b, N, 128)
        scores = torch.softmax(scores, dim=-1) # (b, N, 128)
        
        output = torch.matmul(scores, value.transpose(-2, -1)) # (b, N, dim)
        #output = self.linear(torch.cat([output.transpose(-2, -1),feat],dim=1))

        return (output.transpose(-2, -1)+feat)/2, cons
    
        #print(att.size())
        # agg = torch.sum(dict_feats*w,dim=2,keepdim=True)  # b, dim, n
        # return agg,w
    
class PostProcess(nn.Module):
    def __init__(self, embed_dim=128, upscale=[1,4,4]):
        super(PostProcess, self).__init__()
        #self.mlp_1 = nn.Conv1d(64, 128, 1) #
        #self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[32, 128])
        ##Parametric Surface Constrained Upsampler(self,upscale=4, dim_feat=512)
        up_layers = []
        for i, factor in enumerate(upscale):
            up_layers.append(UpLayer(dim=embed_dim, seed_dim=embed_dim, up_factor=factor, i=i, n_knn=20, radius=1, 
                             interpolate='three', attn_channel=True))
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, seed, fea):
        """
        Args:
            global_shape_fea: Tensor, (b, dim_feat, 1)
            pcd: Tensor, (b, 3, n)
        """
        pred_pcds = [seed.permute(0, 2, 1).contiguous()]
        # Upsample layers
        K_prev = None
        pcd = seed # (B, 3, 256)
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed, fea, K_prev)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())

        return pred_pcds


# 3D completion
@MODELS.register_module()
class UpTrans(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            up_factors: list of int
        """
        super(UpTrans, self).__init__()
        self.num_pred = config.num_pred
        self.use_L2 = config.l2_loss
        self.feat_extractor = UNet(dim_feat=config.dim_feat,num_seeds=config.num_seeds,seed_fea=config.seed_fea, num_dicts=config.num_dicts)
        self.decoder = PostProcess(embed_dim=config.seed_fea,upscale = config.upscales)
        self.build_loss_func()

    def build_loss_func(self):
        if self.use_L2==0:
            self.loss_func = ChamferDistanceL1()
        else:
            self.loss_func = ChamferDistanceL2()
        #self.penalty_func = expansionPenaltyModule()

    def get_loss(self, ret, gt):
        Pc, P3, P1, P2, cons1,cons2 = ret

        cd0 = self.loss_func(Pc, gt)
        cd1 = self.loss_func(P1, gt)
        cd2 = self.loss_func(P2, gt)
        cd3 = self.loss_func(P3, gt)

        loss_all = (1*cd0 + 1*cd1 + 1*cd2 + 1*cd3) * 1e3 
        return loss_all, cd0, cd3
    
    def get_constrain(self, ret):
        Pc, P3, P1,P2, cons1, cons2 = ret
        #P1, P2, r = pcds_pred

        ## orthogonal constraint
        Iden = torch.eye(cons1.size(0),device=cons1.get_device())
        loss_orth1 = 1e3 *torch.nn.functional.mse_loss(cons1, Iden, reduction='mean')

        Iden = torch.eye(cons2.size(0),device=cons2.get_device())
        loss_orth2 = 1e3 *torch.nn.functional.mse_loss(cons2, Iden, reduction='mean')

        #Iden2 = torch.eye(cons2.size(0),device=cons2.get_device())

        #loss_orth1 = 1*torch.nn.functional.mse_loss(cons1, Iden1, reduction='mean')
        #loss_orth2 = 1*torch.nn.functional.mse_loss(cons2, Iden2, reduction='mean')

        return loss_orth1, loss_orth2


    def forward(self, partial_point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """

        #pcd_bnc = point_cloud
        in_pcd = partial_point_cloud.permute(0, 2, 1).contiguous()     

        partial_shape_code,xyz,fea,cons1,cons2 = self.feat_extractor(in_pcd)

        pcd,p1,p2,p3 = self.decoder(xyz, fea)

        return pcd,p3,p1,p2,cons1,cons2