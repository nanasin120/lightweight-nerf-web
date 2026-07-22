import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Callable

class NeRF(nn.Module):
    """
        NeRF 신경말 모델

        Args:
            D (int): 전체 신경망의 레이어 개수 (기본값: 8)
            W (int): 은닉층(Hidden Layer)의 뉴런 개수 (기본값: 256)
            L_embed (int): 위치 인코딩 주파수 대역 개수 (기본값: 6)
    """
    def __init__(
            self, 
            D:int = 8,
            W:int = 256,
            L_embed:int = 6
        ):
        super().__init__()

        self.D = D
        self.W = W
        self.L_embed = L_embed

        input_dim = 3 + 3 * 2 * L_embed

        self.input = nn.Linear(in_features=input_dim, out_features=W, bias=True)

        self.dense = nn.ModuleList()
        for i in range(D-1):
            idx = i + 1 # input이 있을 경우의 idx

            if idx % 5 == 0:
                self.dense.append(nn.Linear(in_features=W + input_dim, out_features=W, bias=True))
            else:
                self.dense.append(nn.Linear(in_features=W, out_features=W, bias=True))
            
        self.output = nn.Linear(in_features=W, out_features=4, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, 
            x:torch.Tensor # [Total_points, input_dim]
        ) -> torch.Tensor:
        """
            신경망의 순전파 연산

            Args:
                x (torch.Tensor): 위치 인코딩이 완료된 3D 샘플 포인트 텐서 
                    Shape: [Total_Points, 3 + (3 * 2 * L_embed)]

            Returns:
                outputs (torch.Tensor): 각 샘플 포인트의 예측된 [R, G, B, Sigma(밀도)] 값
                    Shape: [Total_Points, 4]
        """
        
        inputs = x
        outputs = self.relu(self.input(x))

        for i in range(self.D - 1):
            idx = i + 1

            if idx % 5 == 0:
                outputs = torch.cat([outputs, inputs], dim=-1)

            outputs = self.dense[i](outputs)
            outputs = self.relu(outputs)

        outputs = self.output(outputs)

        return outputs

def positional_encoding(
        x:torch.Tensor,
        L_embed:int=6
    ) -> torch.Tensor:
    """
        위치 정보를 받아 Positional_encoding을 연결해 출력하는 함수

        Args:
            x (torch.Tensor) : 위치 인코딩이 되지 않은 3D 샘플 포인트 텐서
                Shape: [Total_points, 3]
            L_embed (int) : 인코딩을 적용시킬 횟수
        Returns:
            rets (torch.Tensor) : 위치 인코딩이 완료된 3D 샘플 포인트 텐서
                Shape: [Total_points, 3 + (3 * 2 * L_embed)]      
    """
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2 ** i * x))

    rets = torch.cat(rets, dim=-1)
    return rets

def get_rays(
        H:int,
        W:int,
        focal:float,
        c2w:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        카메라의 내부 데이터와 외부 데이터를 받아 각 픽셀을 관통하는 레이의 원점과 방향을 계산하는 함수

        Args:
            H (int): 이미지 높이 (Height)
            W (int): 이미지 너비 (Width)
            focal (float): 초점 거리 (Focal Length)
            c2w (torch.Tensor): Camera-to-World 변환 행렬 
                Shape: [4, 4] 또는 [3, 4]

        Returns:
            rays_o (torch.Tensor): 각 레이의 시작점 (World Coordinate Origin) 
                Shape: [H, W, 3]
            rays_d (torch.Tensor): 각 레이의 방향 벡터 (World Coordinate Direction) 
                Shape: [H, W, 3]
    """
    device = c2w.device

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device), 
        torch.arange(H, dtype=torch.float32, device=device), 
        indexing='xy',)
    
    dirs = torch.stack([
        (i - W * .5) / focal, 
        -(j - H * .5) / focal, 
        -torch.ones_like(i)
        ], dim=-1)

    rays_d = torch.matmul(dirs, c2w[:3, :3].T)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def render_rays(
        network_fn: Callable[[torch.Tensor], torch.Tensor],
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        N_samples: int,
        rand: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        ray에 sampling을 적용한 뒤 각 sample의 rgb, depth, acc를 연산하는 함수

        Args:
            network_fn (Callable): NeRF 모델
            rays_o (torch.Tensor): 각 레이의 시작점
                Shape: [H, W, 3]
            rays_d (torch.Tensor): 각 레이의 방향 벡터
                Shape: [H, W, 3]
            near (float) : 가장 가까운 거리
            far (float) : 가장 먼 거리
            N_samples (int) : 샘플의 개수
            rand (bool) : 샘플링 좌표에 노이즈를 추가하는지 여부

        Returns:
            rgb_map (torch.Tensor): 색상 지도
                Shape: [H, W, 3]
            depth_map (torch.Tensor): 깊이 지도
                Shape: [H, W] 
            acc_map (torch.Tensor): 누적 불투명도 지도
                Shape: [H, W] 
    """
    def batchify(
            fn: Callable[[torch.Tensor], torch.Tensor],
            chunk: int = 1024 * 32 # 한번에 GPU에 넣을 샘플 수
        ):
        def ret(inputs:torch.Tensor) -> torch.Tensor:
            outputs = []
            for i in range(0, inputs.shape[0], chunk):
                outputs.append(fn(inputs[i:i+chunk]))
            outputs = torch.cat(outputs, dim=0)
            return outputs  
        return ret
    
    z_vals = torch.linspace(near, far, N_samples, device=rays_o.device)
    if rand:
        noise = torch.rand(rays_o.shape[:-1] + (N_samples, ), device=rays_o.device)
        z_vals = z_vals + noise * (far - near) / N_samples

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = pts.reshape(-1, 3)
    pts_flat = positional_encoding(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = raw.reshape(pts.shape[:-1] + (4, ))

    # Compute opacities and colors
    sigma_a = torch.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    # Do Volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.full_like(z_vals[..., :1], 1e10)], dim=-1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    
    transmittance_terms = 1. - alpha + 1e-10
    padding = torch.cat([
        torch.ones_like(transmittance_terms[..., :1]),
        transmittance_terms[..., :-1]
    ], dim=-1)
    weights = alpha * torch.cumprod(padding, dim=-1)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map