import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

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

        self.input = nn.Linear(in_features=input_dim, out_features=W, bias=False)

        self.dense = nn.ModuleList()
        for i in range(D-1):
            idx = i + 1 # input이 있을 경우의 idx

            if idx % 5 == 0:
                self.dense.append(nn.Linear(in_features=W + input_dim, out_features=W, bias=False))
            else:
                self.dense.append(nn.Linear(in_features=W, out_features=W, bias=False))
            
        self.output = nn.Linear(in_features=W, out_features=4, bias=False)
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
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32), 
        torch.arange(H, dtype=torch.float32), 
        indexing='xy')
    
    dirs = torch.stack([
        (i - W * .5) / focal, 
        -(j - H * .5) / focal, 
        -torch.ones_like(i)
        ], dim=-1)

    rays_d = torch.matmul(dirs, c2w[:3, :3].T)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d
