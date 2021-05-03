import torch
import torch.nn as nn

class DyReLUA(nn.Module):
    def __init__(self,
                 channels,
                 reduction=4,
                 k=2):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2 * k, 1),
            nn.Sigmoid()
        )

        # default parameter setting
        # lambdaA = 1.0, lambdaB = 0.5;
        # alphaA1 = 1, alphaA2=alphaB1=alphaB2=0
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def forward(self, x):
        coef = self.coef(x)
        coef = 2 * coef - 1
        coef = coef.view(-1, 2 * self.k) * self.lambdas + self.bias

        # activations
        # NCHW --> NCHW1
        x_perm = x.permute(1, 2, 3, 0).unsqueeze(-1)
        # HWNC1 * NK --> HWCNK
        output = x_perm * coef[:, :self.k] + coef[:, self.k:]
        result = torch.max(output, dim=-1)[0].permute(3, 0, 1, 2)
        return result
    
class DyReLUB(nn.Module):
    def __init__(self,
                 channels,
                 reduction=4,
                 k=2):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, 2 * k * channels, 1),
            nn.Sigmoid()
        )

        # default parameter setting
        # lambdaA = 1.0, lambdaB = 0.5;
        # alphaA1 = 1, alphaA2=alphaB1=alphaB2=0
        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def forward(self, x):
        coef = self.coef(x)
        coef = 2 * coef - 1

        # coefficient update
        coef = coef.view(-1, self.channels, 2 * self.k) * self.lambdas + self.bias

        # activations
        # NCHW --> HWNC1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # HWNC1 * NCK --> HWNCK
        output = x_perm * coef[:, :, :self.k] + coef[:, :, self.k:]
        # maxout and HWNC --> NCHW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result
    
class DyReLUC(nn.Module):
    def __init__(self,
                channels,
                reduction=4,
                k=2,
                tau=10,
                gamma=1/3):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k
        self.tau = tau
        self.gamma = gamma

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2 * k * channels, 1),
            nn.Sigmoid()
        )
        self.sptial = nn.Conv2d(channels, 1, 1)

        # default parameter setting
        # lambdaA = 1.0, lambdaB = 0.5;
        # alphaA1 = 1, alphaA2=alphaB1=alphaB2=0
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def forward(self, x):
        N, C, H, W = x.size()
        coef = self.coef(x)
        coef = 2 * coef - 1

        # coefficient update
        coef = coef.view(-1, self.channels, 2 * self.k) * self.lambdas + self.bias

        # spatial
        gamma = self.gamma * H * W
        spatial = self.sptial(x)
        spatial = spatial.view(N, self.channels, -1) / self.tau
        spatial = torch.softmax(spatial, dim=-1) * gamma
        spatial = torch.clamp(spatial, 0, 1).view(N, 1, H, W)

        # activations
        # NCHW --> HWNC1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # HWNC1 * NCK --> HWNCK
        output = x_perm * coef[:, :, :self.k] + coef[:, :, self.k:]

        # permute spatial from NCHW to HWNC1
        spatial = spatial.permute(2, 3, 0, 1).unsqueeze(-1)
        output = spatial * output

        # maxout and HWNC --> NCHW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result