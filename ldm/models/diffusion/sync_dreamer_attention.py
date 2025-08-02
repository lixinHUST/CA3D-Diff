import torch
import torch.nn as nn

from ldm.modules.attention import default, zero_module, checkpoint
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding

class DepthAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, output_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1, 1, bias=False)
        self.to_k = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        self.to_v = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        if output_bias:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1)
        else:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1, bias=False)

    def forward(self, x, context):
        """

        @param x:        b,f0,h,w
        @param context:  b,f1,d,h,w
        @return:
        """
        hn, hd = self.heads, self.dim_head
        b, _, h, w = x.shape
        b, _, d, h, w = context.shape

        q = self.to_q(x).reshape(b,hn,hd,h,w) # b,t,h,w
        k = self.to_k(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w
        v = self.to_v(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w

        sim = torch.sum(q.unsqueeze(3) * k, 2) * self.scale # b,hn,d,h,w
        attn = sim.softmax(dim=2)

        # b,hn,hd,d,h,w * b,hn,1,d,h,w
        out = torch.sum(v * attn.unsqueeze(2), 3) # b,hn,hd,h,w
        out = out.reshape(b,hn*hd,h,w)
        return self.to_out(out)


class DepthTransformer(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, checkpoint=True):
        super().__init__()
        inner_dim = n_heads * d_head
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1),
            nn.GroupNorm(8, inner_dim),
            nn.SiLU(True),
        )
        self.proj_context = nn.Sequential(
            nn.Conv3d(context_dim, context_dim, 1, 1, bias=False), # no bias
            nn.GroupNorm(8, context_dim),
            nn.ReLU(True), # only relu, because we want input is 0, output is 0
        )
        self.depth_attn = DepthAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, context_dim=context_dim, output_bias=False)  # is a self-attention if not self.disable_self_attn
        self.proj_out = nn.Sequential(
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            zero_module(nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)),
        )
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        x_in = x
        x = self.proj_in(x)
        context = self.proj_context(context)
        x = self.depth_attn(x, context)
        x = self.proj_out(x) + x_in
        return x

class DepthAttention2D(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, output_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1, 1, bias=False)
        self.to_k = nn.Conv2d(context_dim, inner_dim, 1, 1, bias=False)
        self.to_v = nn.Conv2d(context_dim, inner_dim, 1, 1, bias=False)

        self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1, bias=output_bias)

    def forward(self, x, context):
        """
        :param x:       (B, C, H, W)
        :param context: (B, C, H, W)
        :return:        (B, C, H, W)
        """
        b, _, h, w = x.shape
        hn, hd = self.heads, self.dim_head

        q = self.to_q(x).reshape(b, hn, hd, h * w)
        k = self.to_k(context).reshape(b, hn, hd, h * w)
        v = self.to_v(context).reshape(b, hn, hd, h * w)

        sim = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale  # attention weights
        attn = sim.softmax(dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)  # weighted sum
        out = out.reshape(b, hn * hd, h, w)
        return self.to_out(out)

class DepthTransformer2D(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, checkpoint=True):
        super().__init__()
        inner_dim = n_heads * d_head
        context_dim = default(context_dim, dim)

        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1),
            nn.GroupNorm(8, inner_dim),
            nn.SiLU(True),
        )

        self.proj_context = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, 1, 1, bias=False),
            nn.GroupNorm(8, context_dim),
            nn.ReLU(True),
        )

        self.depth_attn = DepthAttention2D(
            query_dim=inner_dim, 
            heads=n_heads, 
            dim_head=d_head, 
            context_dim=context_dim, 
            output_bias=False
        )

        self.proj_out = nn.Sequential(
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            zero_module(nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)),
        )

        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        x_in = x
        x = self.proj_in(x)
        context = self.proj_context(context)
        x = self.depth_attn(x, context)
        x = self.proj_out(x) + x_in
        return x

class DepthWiseAttention(UNetModel):
    def __init__(self, volume_dims=(5,16,32,64), *args, **kwargs):
        super().__init__(*args, **kwargs)
        # num_heads = 4
        model_channels = kwargs['model_channels']
        channel_mult = kwargs['channel_mult']
        d0,d1,d2,d3 = volume_dims

        # 4
        ch = model_channels*channel_mult[2]
        self.middle_conditions = DepthTransformer2D(ch, 4, d3 // 2, context_dim=d3)

        self.output_conditions=nn.ModuleList()
        self.output_b2c = {3:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8}
        # 8
        ch = model_channels*channel_mult[2]
        self.output_conditions.append(DepthTransformer2D(ch, 4, d2 // 2, context_dim=d2)) # 0
        self.output_conditions.append(DepthTransformer2D(ch, 4, d2 // 2, context_dim=d2)) # 1
        # 16
        self.output_conditions.append(DepthTransformer2D(ch, 4, d1 // 2, context_dim=d1)) # 2
        ch = model_channels*channel_mult[1]
        self.output_conditions.append(DepthTransformer2D(ch, 4, d1 // 2, context_dim=d1)) # 3
        self.output_conditions.append(DepthTransformer2D(ch, 4, d1 // 2, context_dim=d1)) # 4
        # 32
        self.output_conditions.append(DepthTransformer2D(ch, 4, d0 // 2, context_dim=d0)) # 5
        ch = model_channels*channel_mult[0]
        self.output_conditions.append(DepthTransformer2D(ch, 4, d0 // 2, context_dim=d0)) # 6
        self.output_conditions.append(DepthTransformer2D(ch, 4, d0 // 2, context_dim=d0)) # 7
        self.output_conditions.append(DepthTransformer2D(ch, 4, d0 // 2, context_dim=d0)) # 8

    def forward(self, x, timesteps=None, context=None, y=None,source_dict=None, **kwargs):

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            y=y.squeeze(1)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        context_32=context
        context_16=self.downsample_pool(context_32)
        context_8=self.downsample_pool(context_16)
        context_4=self.downsample_pool(context_8)
        context_dict={'4':context_4,'8':context_8,'16':context_16,'32':context_32}
        
        for index, module in enumerate(self.input_blocks):
            context = self._select_context(h, context_dict)
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)
        h = self.middle_conditions(h, context=source_dict[h.shape[-1]])

        for index, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            context = self._select_context(h, context_dict)
            h = module(h, emb, context)
            if index in self.output_b2c:
                layer = self.output_conditions[self.output_b2c[index]]
                h = layer(h, context=source_dict[h.shape[-1]])

        h = h.type(x.dtype)
        return self.out(h)

    def get_trainable_parameters(self):
        paras = [para for para in self.middle_conditions.parameters()] + [para for para in self.output_conditions.parameters()]
        return paras
