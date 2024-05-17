#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Pingping Cai

import torch
import torch.nn as nn
import math
from extensions.chamfer_dist import ChamferDistanceL1,ChamferDistanceL1_split
#from extensions.expansion_penalty.expansion_penalty_module import expansionPenaltyModule
from models.pointnet import PointNet_SA_Module_KNN, MLP_Res, UpTransformer, Transformer
from models.upsample import PSCU

from .build import MODELS


class UNet(nn.Module):
    def __init__(self, dim_feat=384, num_seeds = 1024, num_dicts=128):
        '''
        Extract information from partial point cloud
        '''
        super(UNet, self).__init__()
        self.num_seed = num_seeds
        self.sa_module_1 = PointNet_SA_Module_KNN(num_seeds//2, 16, 3, [32, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(num_seeds//8, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        #self.sa_module_3 = PointNet_SA_Module_KNN(32, 16, 256, [256, 384], group_all=False, if_bn=False, if_idx=True)
        #self.transformer_3 = Transformer(384, dim=64)
        self.sa_module_4 = PointNet_SA_Module_KNN(None, None, 256, [384, dim_feat], group_all=True, if_bn=False)
        
        self.ps_0 = nn.ConvTranspose1d(dim_feat, 256, num_seeds//16, bias=True)
        self.ps_1 = nn.ConvTranspose1d(256, 256, 2,2, bias=True)
        self.ps_2 = nn.ConvTranspose1d(256, 128, 2,2, bias=True)
        #self.seedgen = SeedGenerator(feat_dim=dim_feat, seed_dim=256, n_knn=20, factor=2, attn_channel=True)

        #self.uptrans_1 = UpTransformer(256, 256, dim=64, n_knn=16, use_upfeat=False, attn_channel=True, up_factor=2, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 256, hidden_dim=256, out_dim=256)
        #self.uptrans_2 = UpTransformer(256, 128, dim=64, n_knn=16, use_upfeat=False, attn_channel=True, up_factor=2, scale_layer=None)
        #self.mlp_2 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        
        self.refine_1 = FeaRefine(dim=256,hidd_dim=64,num_dicts=num_dicts)
        self.refine_2 = FeaRefine(dim=128,hidd_dim=64,num_dicts=num_dicts//2)
        
        #self.refine_1 = FeaRefine(dim=256,hidd_dim=64)
        #self.refine_2 = FeaRefine(dim=128,hidd_dim=64)

        #self.refine_1 = MLP_Res(in_dim=256, hidden_dim=256, out_dim=256)
        #self.refine_2 = MLP_Res(in_dim=128, hidden_dim=128, out_dim=128)

        #self.ps_3 = nn.ConvTranspose1d(64, 32, 2, bias=True)
        #self.mlp_1 = MLP_Res(in_dim=dim_feat + 256, hidden_dim=256, out_dim=256)
        #self.mlp_2 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=64)
        #self.pos_mlp0 = MLP_Res(in_dim=256, hidden_dim=64, out_dim=3)
        #self.pos_mlp1 = MLP_Res(in_dim=256, hidden_dim=64, out_dim=3)
        self.mlp_3 = MLP_Res(in_dim=128, hidden_dim=32, out_dim=3)

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
        #upconv 1
        u0_fea = self.ps_0(l4_fea)  # (b, 256, 128)
        u0_fea = self.mlp_1(torch.cat([u0_fea, l4_fea.repeat((1, 1, u0_fea.size(2)))], 1))
        #u1_xyz = self.pos_mlp1(u1_fea)
        #u1_xyz,u1_fea = self.seedgen(l4_fea,l2_xyz, l2_fea)
        #u1_fea = self.uptrans_1(u1_xyz, u1_fea, u1_fea, upfeat=None)
        #u1_fea = self.mlp_1(torch.cat([u1_fea, l4_fea.repeat((1, 1, u1_fea.size(2)))], 1))
        #u1_fea = self.refine_1(u1_fea)
        u1_fea = self.ps_1(u0_fea)
        
        u1_fea,cons1 = self.refine_1(u1_fea, surge=False)
        #u1_fea = u1_fea+self.mlp_1(torch.cat([u1_fea, l4_fea.repeat((1, 1, u1_fea.size(2)))], 1)) # (b, 256, 128)
        #u1_xyz = self.pos_mlp1(u1_fea)
        
        # skip concat
        u1_fea = torch.cat([l2_fea,u1_fea],dim=2)  # (b, 256, 256)
        #u1_xyz = torch.cat([l2_xyz,u1_xyz],dim=2)
        
        #upconv 2
        u2_fea = self.ps_2(u1_fea)  # (b, 64, 512)
        #u2_fea = self.uptrans_2(u1_xyz, u1_fea, u1_fea, upfeat=None)
        #u2_fea = self.mlp_2(torch.cat([u2_fea, l4_fea.repeat((1, 1, u2_fea.size(2)))], 1)) # (b, 64, 512)
        #u2_fea = self.refine_2(u2_fea)
        u2_fea,cons2 = self.refine_2(u2_fea)
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

        # fea_mem1 = torch.nn.functional.normalize(self.dict_embed1.weight,dim=0)
        # fea_mem2 = torch.nn.functional.normalize(self.dict_embed2.weight,dim=0)
        #fea_mem1 = self.dict_embed1.weight
        #fea_mem2 = self.dict_embed2.weight
        # cons1 = fea_mem1.transpose(0,1) @ fea_mem1
        # cons2 = fea_mem2.transpose(0,1) @ fea_mem2

        return l4_fea, u2_xyz, u2_fea,u2_xyz, cons1, cons2

class SeedGenerator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=20, factor=2, attn_channel=True):
        super(SeedGenerator, self).__init__()
        self.uptrans = UpTransformer(256, 128, dim=64, n_knn=n_knn, use_upfeat=False, attn_channel=attn_channel, up_factor=factor, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor (B, feat_dim, 1)
            patch_xyz: (B, 3, 128)
            patch_feat: (B, seed_dim, 128)
        """
        x1 = self.uptrans(patch_xyz, patch_feat, patch_feat, upfeat=None)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        completion = self.mlp_4(x3)  # (B, 3, 256)
        return completion, x3

class FeaRefine(nn.Module):
    def __init__(self, dim=512, hidd_dim=64, num_dicts=128):
        super(FeaRefine, self).__init__()
        self.num_dicts = num_dicts
        self.dict_embed = nn.Parameter(torch.randn(dim, num_dicts))
        self.query = nn.Conv1d(dim, hidd_dim, 1)
        self.key = nn.Conv1d(dim, hidd_dim, 1)
    
    def forward(self, feat, surge=True):
        """
        Args:
            feat: Tensor (b, dim_feat, N)
        """
        B, C, N = feat.size()
        dict_fea = self.dict_embed  # (dim, num_dicts)
        feature_memory = torch.nn.functional.normalize(dict_fea, dim=0)
        cons = feature_memory.transpose(0, 1) @ feature_memory

        dict_fea = dict_fea.unsqueeze(0).repeat(B, 1, 1)
        
        q = self.query(feat)  # (b, 64, N)
        value = dict_fea
        k = self.key(value)  # (b, 64, num_dicts)
        
        d_k = k.size(1)
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k)  # (b, N, num_dicts)
        scores = torch.softmax(scores, dim=-1)  # (b, N, num_dicts)
        
        output = torch.matmul(scores, value.transpose(-2, -1))  # (b, N, dim)
        surg = (output.transpose(-2, -1) + feat) / 2

        if surge:
            surg = dict_fea
        return surg, scores
      
class FeaRefine1(nn.Module):
    def __init__(self, dim=512, hidd_dim=64, num_dicts=128):
        super(FeaRefine1, self).__init__()
        self.num_dicts = num_dicts
        #self.merge = nn.Conv1d(dim+dim, dim, 1,bias=False)
        self.query = nn.Conv1d(dim, hidd_dim, 1)
        self.key = nn.Conv1d(dim, hidd_dim, 1)
        #self.value = nn.Conv1d(512+dim, dim, 1)
    
    def forward(self, feat,dict_embed,global_fea):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        B = feat.size(0)
        C,N = dict_embed.weight.size()
        #print(feat.size())
        dict_fea = dict_embed.weight.unsqueeze(0).repeat(B,1,1) # (1, dim, 128)
        #value = self.value(torch.cat([global_fea.repeat(1,1,N),dict_fea],dim=1))
        #shape_code = coarse_shapecode.repeat(1,1,128)
        #cons = dict_fea.transpose(0,1) @ dict_fea
        #Iden = torch.eye(self.num_dict,device='cuda')

        #value = value.repeat(B,1,1)
        
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
        output = self.merge(torch.cat([output.transpose(-2, -1),feat],dim=1))

        return output
        #return (output.transpose(-2, -1)+feat)/2
    
        #print(att.size())
        # agg = torch.sum(dict_feats*w,dim=2,keepdim=True)  # b, dim, n
        # return agg,w


class FoldingNet(nn.Module):
    def __init__(self, encoder_channel= 128, num_pred=256):
        super().__init__()
        self.num_pred = num_pred
        self.encoder_channel = encoder_channel
        self.grid_size = int(pow(self.num_pred,0.5) + 0.5)

        #self.mlp_1 = nn.Conv1d(64, encoder_channel, 1) #
        
        self.folding1 = nn.Sequential(
            nn.Conv2d(self.encoder_channel + 2, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv2d(self.encoder_channel + 3, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, 1),
        )

        a = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda() # 1 2 N


    def forward(self, seed, x):
        num_sample = self.grid_size * self.grid_size
        bs,c,N = x.size()
        #print(x.size())
        #feat_1 = self.mlp_1(x)
        features = x.view(bs, self.encoder_channel, N, 1).expand(bs, self.encoder_channel, N, num_sample)
        grid = self.folding_seed.view(1, 2, 1, num_sample).expand(bs, 2, N, num_sample).to(x.device)
        seed_xyz = seed.view(bs, 3, N, 1).expand(bs, 3, N, num_sample)

        x = torch.cat([grid, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        out1 = fd1+seed_xyz
        out2 = fd2+seed_xyz
        #print(fd2.size())
        surg = fd2[:,:,512:,:]
        #print(surg.size())
        out1 = out1.view(bs,3,-1)
        out2 = out2.view(bs,3,-1)
        surg = surg.transpose(2,1).contiguous()
        #print (surg.size())
        return seed.transpose(2,1).contiguous(), out1.transpose(2,1).contiguous() , out2.transpose(2,1).contiguous(), surg
        # return fd2.transpose(2,1).contiguous()

# 3D completion
@MODELS.register_module()
class DictFold(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            up_factors: list of int
        """
        super(DictFold, self).__init__()
        self.num_pred = 256 #config.num_pred // config.num_seeds
        self.sparse_expansion_lambda = config.sparse_expansion_lambda
        self.dense_expansion_lambda = config.dense_expansion_lambda
        self.feat_extractor = UNet(dim_feat=config.dim_feat,num_seeds=config.num_seeds, num_dicts=config.num_dicts)
        self.decoder = FoldingNet(encoder_channel=128,num_pred= self.num_pred)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        self.loss_split = ChamferDistanceL1_split()
        #self.penalty_func = expansionPenaltyModule()

    def get_loss(self, ret, gt):
        Pc, P2,P0, P1,cons1, cons2 = ret
        cdc = self.loss_func(Pc, gt)
        cd0,_ = self.loss_split(P0, gt)
        cd1 = self.loss_func(P1, gt)
        cd2 = self.loss_func(P2, gt)

        loss_all = (1*cdc + 0*cd0 + 1*cd1 + 1*cd2 ) * 1e3 
        return loss_all, cdc, cd2
    
    def get_constrain(self, ret):
        Pc,p0,p1,p2, cons1, cons2 = ret
        #P1, P2, r = pcds_pred
        
        ## orthogonal constraint
        Iden1 = torch.eye(cons1.size(0),device=cons1.get_device())
        Iden2 = torch.eye(cons2.size(0),device=cons2.get_device())
        loss_orth1 = 1*torch.nn.functional.mse_loss(cons1, Iden1, reduction='mean')
        loss_orth2 = 1*torch.nn.functional.mse_loss(cons2, Iden2, reduction='mean')

        return loss_orth1, loss_orth2


    def forward(self, partial_point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """

        #pcd_bnc = point_cloud
        in_pcd = partial_point_cloud.permute(0, 2, 1).contiguous()     

        partial_shape_code,xyz,fea,p0,score1,score2 = self.feat_extractor(in_pcd)

        pcd,p1,p2, surg = self.decoder(xyz, fea)
        #print(pcd.size())
        #print(p2.size())
        return pcd,p2,p0,p1,score1,score2,surg