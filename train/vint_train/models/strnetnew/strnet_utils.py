import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from efficientnet_pytorch import EfficientNet


class STRNetNew_Extractor(nn.Module):
    def __init__(
        self,
        temporal_method: str = "stgcn",
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        gnn_layers: int = 3,
        K_list: list = [8,4,2,1],
    ) -> None:
        super().__init__()
        self.temporal_method = temporal_method
        self.context_size = context_size
        self.obs_encoding_size = obs_encoding_size
        
        self._init_encoders(obs_encoder)
        
        
        self.st_gnn = nn.Sequential(*[
            SpatioTemporalGrapher(
                in_channels=obs_encoding_size,
                K=K_list[i],
                temporal_length=context_size+2
            ) for i in range(gnn_layers)
        ])

        self.final_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_encoding_size * (context_size + 2), obs_encoding_size),
            nn.BatchNorm1d(obs_encoding_size),
            nn.GELU()
        )

    def _init_encoders(self, obs_encoder):
        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
        self.obs_adapter = self._build_adapter(
            self.obs_encoder._fc.in_features, 
            self.obs_encoding_size
        )
        
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.goal_adapter = self._build_adapter(
            self.goal_encoder._fc.in_features,
            self.obs_encoding_size
        )

    def _build_adapter(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=out_dim),
            nn.GELU()
        )

    def _extract_base_features(self, obs_img, goal_img):
        B, C, H, W = obs_img.shape
        T = self.context_size + 1
        
        obs_img = obs_img.view(B*T, 3, H, W)
        obs_feats = self.obs_encoder.extract_features(obs_img)
        obs_feats = self.obs_encoder._avg_pooling(obs_feats)
        obs_feats = self.obs_adapter(obs_feats)
        obs_feats = obs_feats.view(B, T, self.obs_encoding_size, 1, 1)
        
        current_frame = obs_img.view(B, T, 3, H, W)[:, -1]
        goal_input = torch.cat([
            current_frame, 
            F.interpolate(goal_img, size=(H, W), mode='bilinear')
        ], dim=1)
        goal_feats = self.goal_encoder.extract_features(goal_input)
        goal_feats = self.goal_encoder._avg_pooling(goal_feats)
        goal_feats = self.goal_adapter(goal_feats)
        goal_feats = goal_feats.unsqueeze(1)
        
        return obs_feats, goal_feats

    def forward(self, obs_img, goal_img, **kwargs):
        obs_feats, goal_feats = self._extract_base_features(obs_img, goal_img)
        
        st_cube = torch.cat([obs_feats, goal_feats], dim=1)
        
        temporal_out = self.st_gnn(st_cube)
        
        return self.final_proj(temporal_out.view(temporal_out.size(0), -1))


class SpatioTemporalGrapher(nn.Module):
    def __init__(self, in_channels, K, temporal_length):
        super().__init__()
        self.A = int(in_channels ** 0.5)
        assert self.A * self.A == in_channels, f"in_channels必须为平方数，当前为{in_channels}"

        self.temporal_length = temporal_length

        self.spatial_grapher = Grapher(in_channels=1, K=K)

        self.tsm = TSMModule(shift_ratio=0.125, in_channels=in_channels)

        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(1),
            nn.GELU()
        )

        self.mr_conv = DynamicMRConv3D(in_channels=1, K=K)

        self.fusion = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.GELU()
        )

    def forward(self, x):
        B, T, D, _, _ = x.shape
        A = self.A
        assert D == A * A, f"Input channel D must match A*A, got D={D}, A={A}"

        x = x.view(B, T, 1, A, A)

        x_spatial = x.view(B * T, 1, A, A)
        spatial_out = self.spatial_grapher(x_spatial)
        spatial_out = spatial_out.view(B, T, 1, A, A)

        spatial_out_flat = spatial_out.view(B, T, D, 1, 1)

        tsm_out = self.tsm(spatial_out_flat)

        tsm_out = tsm_out.view(B, T, 1, A, A)

        temporal_input = tsm_out.permute(0, 2, 1, 3, 4)

        temporal_out = self.temporal_conv(temporal_input)
        mr_out = self.mr_conv(temporal_out)

        fused = self.fusion(torch.cat([temporal_input, mr_out], dim=1))

        fused = fused.permute(0, 2, 1, 3, 4)
        return fused.view(B, T, D, 1, 1)




class DynamicMRConv3D(nn.Module):
    def __init__(self, in_channels, K):
        super().__init__()
        self.K = K
        self.scale_weights = nn.Parameter(torch.ones(K))
        
        self.nn = nn.Sequential(
            nn.Conv3d(2, 1, 1),
            nn.BatchNorm3d(1),
            nn.GELU()
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        
        upsampled_list = []
        for i in range(self.K):
            h = max(H // (2 ** i), 4)
            w = max(W // (2 ** i), 4)
            
            pooled = F.adaptive_avg_pool3d(x, (T, h, w))
            
            rolled = torch.roll(pooled, shifts=(h//2, w//2), dims=(3,4))
            
            upsampled = F.interpolate(rolled, (T, H, W), mode='trilinear')
            upsampled_list.append(upsampled)
        
        upsampled_stack = torch.stack(upsampled_list, dim=1)
        x_expanded = x.unsqueeze(1)
        
        similarity = F.cosine_similarity(x_expanded, upsampled_stack, dim=2)
        mask = (similarity > similarity.mean(dim=(2,3,4), keepdim=True)).float()
        
        weighted_diff = (self.scale_weights.view(1, self.K, 1, 1, 1, 1) * 
                        (upsampled_stack - x_expanded) * 
                        mask.unsqueeze(2))
        
        x_j = weighted_diff.sum(dim=1)
        
        return self.nn(torch.cat([x, x_j], dim=1))

class ConditionalPositionEncoding(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x

class Grapher(nn.Module):
    def __init__(self, in_channels, K):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = SoftEdgeDynamicMRConv4d(in_channels * 2, in_channels, K=self.K)  
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

       
    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)

        return x

class SoftEdgeDynamicMRConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, K, temperature=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.K = K
        self.temperature = temperature

    def compute_soft_weighted_diff(self, x, dim):
        B, C, H, W = x.size()
        max_shift = H if dim == 2 else W
        soft_diff = torch.zeros_like(x)

        weights = []
        diffs = []

        for s in range(self.K, max_shift, self.K):
            rolled = torch.roll(x, shifts=s, dims=dim)
            diff = rolled - x
            dist = torch.abs(diff).sum(dim=1, keepdim=True)
            sim = torch.exp(-dist / self.temperature)
            weights.append(sim)
            diffs.append(diff * sim)

        if len(diffs) == 0:
            return soft_diff

        stacked_diffs = torch.stack(diffs, dim=0)
        stacked_weights = torch.stack(weights, dim=0)

        weight_sum = stacked_weights.sum(dim=0) + 1e-6
        soft_agg = stacked_diffs.sum(dim=0) / weight_sum
        return soft_agg

    def forward(self, x):
        x_diff_h = self.compute_soft_weighted_diff(x, dim=2)
        x_diff_w = self.compute_soft_weighted_diff(x, dim=3)
        x_j = torch.maximum(x_diff_h, x_diff_w)
        x_cat = torch.cat([x, x_j], dim=1)
        return self.conv(x_cat)

class TSMModule(nn.Module):
    def __init__(self, shift_ratio=0.125, in_channels=512):
        super().__init__()
        self.shift_ratio = shift_ratio
        self.channel_split = [int(in_channels * shift_ratio)] * 4
        self.channel_split[-1] = in_channels - sum(self.channel_split[:-1])
        
        self.temporal_fusion = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.GroupNorm(32, in_channels//2),
            nn.GELU(),
            nn.Conv3d(in_channels//2, in_channels, kernel_size=(1,1,1)),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_shifted = torch.zeros_like(x)
        
        split_chs = torch.split(x, self.channel_split, dim=2)
        
        x_shifted[:, 1:, :self.channel_split[0]] = split_chs[0][:, :-1]
        x_shifted[:, :-1, self.channel_split[0]:sum(self.channel_split[:2])] = split_chs[1][:, 1:]
        x_shifted[:, 1:-1, sum(self.channel_split[:2]):sum(self.channel_split[:3])] = \
            (split_chs[2][:, :-2] + split_chs[2][:, 2:]) / 2
        
        fused = self.temporal_fusion(x_shifted.permute(0,2,1,3,4))
        return x + fused.permute(0,2,1,3,4)

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module