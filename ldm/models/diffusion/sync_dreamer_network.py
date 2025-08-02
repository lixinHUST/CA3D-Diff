import torch
import torch.nn as nn

class Image2DResBlockWithTV(nn.Module):
    def __init__(self, dim, tdim):
        super().__init__()
        norm = lambda c: nn.GroupNorm(8, c)
        self.time_embed = nn.Conv2d(tdim, dim, 1, 1)
        self.conv = nn.Sequential(
            norm(dim),
            nn.SiLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            norm(dim),
            nn.SiLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, x, t):
        return x+self.conv(x+self.time_embed(t))


class NoisyTargetViewEncoder(nn.Module):
    def __init__(self, time_embed_dim, run_dim=16, output_dim=8):
        super().__init__()

        self.init_conv = nn.Conv2d(4, run_dim, 3, 1, 1)
        self.out_conv0 = Image2DResBlockWithTV(run_dim, time_embed_dim)
        self.out_conv1 = Image2DResBlockWithTV(run_dim, time_embed_dim)
        self.out_conv2 = Image2DResBlockWithTV(run_dim, time_embed_dim)
        self.final_out = nn.Sequential(
            nn.GroupNorm(8, run_dim),
            nn.SiLU(True),
            nn.Conv2d(run_dim, output_dim, 3, 1, 1)
        )

    def forward(self, x, t):
        B, DT = t.shape
        t = t.view(B, DT, 1, 1)
        x = self.init_conv(x)
        x = self.out_conv0(x, t)
        x = self.out_conv1(x, t)
        x = self.out_conv2(x, t)
        x = self.final_out(x)
        return x

class SpatialUpTimeBlock(nn.Module):
    def __init__(self, x_in_dim, t_in_dim, out_dim):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv3d(t_in_dim, x_in_dim, 1, 1)  # 16
        self.norm = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.ConvTranspose3d(x_in_dim, out_dim, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.norm(x)))

class SpatialTimeBlock(nn.Module):
    def __init__(self, x_in_dim, t_in_dim, out_dim, stride):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv3d(t_in_dim, x_in_dim, 1, 1)  # 16
        self.bn = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.Conv3d(x_in_dim, out_dim, 3, stride=stride, padding=1)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.bn(x)))

class SpatialTime3DNet(nn.Module):
        def __init__(self, time_dim=256, input_dim=128, dims=(32, 64, 128, 256)):
            super().__init__()
            d0, d1, d2, d3 = dims
            dt = time_dim

            self.init_conv = nn.Conv3d(input_dim, d0, 3, 1, 1)  # 32
            self.conv0 = SpatialTimeBlock(d0, dt, d0, stride=1)

            self.conv1 = SpatialTimeBlock(d0, dt, d1, stride=2)
            self.conv2_0 = SpatialTimeBlock(d1, dt, d1, stride=1)
            self.conv2_1 = SpatialTimeBlock(d1, dt, d1, stride=1)

            self.conv3 = SpatialTimeBlock(d1, dt, d2, stride=2)
            self.conv4_0 = SpatialTimeBlock(d2, dt, d2, stride=1)
            self.conv4_1 = SpatialTimeBlock(d2, dt, d2, stride=1)

            self.conv5 = SpatialTimeBlock(d2, dt, d3, stride=2)
            self.conv6_0 = SpatialTimeBlock(d3, dt, d3, stride=1)
            self.conv6_1 = SpatialTimeBlock(d3, dt, d3, stride=1)

            self.conv7 = SpatialUpTimeBlock(d3, dt, d2)
            self.conv8 = SpatialUpTimeBlock(d2, dt, d1)
            self.conv9 = SpatialUpTimeBlock(d1, dt, d0)

        def forward(self, x, t):
            B, C = t.shape
            t = t.view(B, C, 1, 1, 1)

            x = self.init_conv(x)
            conv0 = self.conv0(x, t)

            x = self.conv1(conv0, t)
            x = self.conv2_0(x, t)
            conv2 = self.conv2_1(x, t)

            x = self.conv3(conv2, t)
            x = self.conv4_0(x, t)
            conv4 = self.conv4_1(x, t)

            x = self.conv5(conv4, t)
            x = self.conv6_0(x, t)
            x = self.conv6_1(x, t)

            x = conv4 + self.conv7(x, t)
            x = conv2 + self.conv8(x, t)
            x = conv0 + self.conv9(x, t)
            return x

class FrustumTVBlock(nn.Module):
    def __init__(self, x_dim, t_dim, out_dim, stride):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv3d(t_dim, x_dim, 1, 1) # 16
        self.bn = norm_act(x_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.Conv3d(x_dim, out_dim, 3, stride=stride, padding=1)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.bn(x)))

class FrustumTVUpBlock(nn.Module):
    def __init__(self, x_dim, t_dim, out_dim):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv3d(t_dim, x_dim, 1, 1) # 16
        self.norm = norm_act(x_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.ConvTranspose3d(x_dim, out_dim, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.norm(x)))

class FrustumTV3DNet(nn.Module):
    def __init__(self, in_dim, t_dim, dims=(32, 64, 128, 256)):
        super().__init__()
        self.conv0 = nn.Conv3d(in_dim, dims[0], 3, 1, 1) # 32

        self.conv1 = FrustumTVBlock(dims[0], t_dim, dims[1], 2)
        self.conv2 = FrustumTVBlock(dims[1], t_dim, dims[1], 1)

        self.conv3 = FrustumTVBlock(dims[1], t_dim, dims[2], 2)
        self.conv4 = FrustumTVBlock(dims[2], t_dim, dims[2], 1)

        self.conv5 = FrustumTVBlock(dims[2], t_dim, dims[3], 2)
        self.conv6 = FrustumTVBlock(dims[3], t_dim, dims[3], 1)

        self.up0 = FrustumTVUpBlock(dims[3], t_dim, dims[2])
        self.up1 = FrustumTVUpBlock(dims[2], t_dim, dims[1])
        self.up2 = FrustumTVUpBlock(dims[1], t_dim, dims[0])

    def forward(self, x, t):
        B,DT = t.shape
        t = t.view(B,DT,1,1,1)

        b, _, d, h, w = x.shape
        x0 = self.conv0(x)
        x1 = self.conv2(self.conv1(x0, t), t)
        x2 = self.conv4(self.conv3(x1, t), t)
        x3 = self.conv6(self.conv5(x2, t), t)

        x2 = self.up0(x3, t) + x2
        x1 = self.up1(x2, t) + x1
        x0 = self.up2(x1, t) + x0
        return {w: x0, w//2: x1, w//4: x2, w//8: x3}

import torch
import torch.nn as nn

class FrustumTVBlock2D(nn.Module):
    def __init__(self, x_dim, t_dim, out_dim, stride):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv2d(t_dim, x_dim, 1, 1)  # (B, t_dim, 1, 1) → (B, x_dim, H, W)
        self.bn = norm_act(x_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.Conv2d(x_dim, out_dim, 3, stride=stride, padding=1)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.bn(x)))

class FrustumTVUpBlock2D(nn.Module):
    def __init__(self, x_dim, t_dim, out_dim):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv2d(t_dim, x_dim, 1, 1)
        self.norm = norm_act(x_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.ConvTranspose2d(x_dim, out_dim, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.norm(x)))

class FrustumTV2DNet(nn.Module):
    def __init__(self, in_dim, t_dim, dims=(32, 64, 128, 256)):
        super().__init__()
        self.conv0 = nn.Conv2d(in_dim, dims[0], 3, 1, 1)

        self.conv1 = FrustumTVBlock2D(dims[0], t_dim, dims[1], 2)
        self.conv2 = FrustumTVBlock2D(dims[1], t_dim, dims[1], 1)

        self.conv3 = FrustumTVBlock2D(dims[1], t_dim, dims[2], 2)
        self.conv4 = FrustumTVBlock2D(dims[2], t_dim, dims[2], 1)

        self.conv5 = FrustumTVBlock2D(dims[2], t_dim, dims[3], 2)
        self.conv6 = FrustumTVBlock2D(dims[3], t_dim, dims[3], 1)

        self.up0 = FrustumTVUpBlock2D(dims[3], t_dim, dims[2])
        self.up1 = FrustumTVUpBlock2D(dims[2], t_dim, dims[1])
        self.up2 = FrustumTVUpBlock2D(dims[1], t_dim, dims[0])

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B, t_dim)
        B, C, H, W = x.shape
        B,DT = t.shape
        t = t.view(B,DT,1,1)
        
        x0 = self.conv0(x)
        x1 = self.conv2(self.conv1(x0, t), t)
        x2 = self.conv4(self.conv3(x1, t), t)
        x3 = self.conv6(self.conv5(x2, t), t)

        x2 = self.up0(x3, t) + x2
        x1 = self.up1(x2, t) + x1
        x0 = self.up2(x1, t) + x0
        return {
            W: x0,
            W // 2: x1,
            W // 4: x2,
            W // 8: x3
        }
