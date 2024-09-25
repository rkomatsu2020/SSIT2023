from munch import Munch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

from SSIT.config import netD_params

class ViTDiscriminator(nn.Module):
    def __init__(self, input_ch: int, **kargs):
        super().__init__()
        img_size = kargs["img_size"]
        hidden_dim_list = netD_params["hidden_dim"]
        head_num = netD_params["head_num"]
        patch_size = netD_params["patch_size"]
        domain_num = kargs["domain_num"]

        patch_num = img_size[0]//patch_size * img_size[1]//patch_size
        patch_elements = 3 * patch_size * patch_size


        self.patch = Patches(patch_size=patch_size)
        self.patch_emb = PatchEmbedding(patch_elements=patch_elements, patch_num=patch_num, emb_dim=hidden_dim_list[0])

        self.trans_block = nn.ModuleDict()
        for i in range(1, len(hidden_dim_list)):
            self.trans_block["net{}".format(i)] = Transformer(input_dim=hidden_dim_list[i-1], 
                                                              output_dim=hidden_dim_list[i],
                                                              num_heads=head_num)
            
        self.last_block = Transformer(input_dim=hidden_dim_list[-1],
                                      output_dim=hidden_dim_list[-1],
                                      num_heads=head_num)
        self.last_logits = nn.Linear(hidden_dim_list[-1], 1, bias=False)

    def forward(self, x, **kargs):
        B = x.size()[0]
        c = kargs["domain"]

        h = self.patch(x)
        h = self.patch_emb(h)
        # Hidden Transformar Blocks
        for k in self.trans_block.keys():
            h = self.trans_block[k](h)
        # Last Transformer block
        out = self.last_block(h)
        out = self.last_logits(out)
        # Select by c
        #idx = torch.LongTensor(range(c.size(0))).to(x.device)
        #out = out[idx, c]

        return {"patch":F.sigmoid(out)}

class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()

        self.patch_size = patch_size

    def forward(self, x):
        '''
        x:[B, C, H, W]
        out: [B, N, C*p1*p2] (N: H//p1*W//p2)
        '''
        B, C, H, W = x.size()
        p = x.unfold(2, self.patch_size, self.patch_size) # B, C, H, W -> B, C, H_p1, W, H_p2
        p = p.unfold(3, self.patch_size, self.patch_size) # B, C, H_p1, W, H_p2 -> B, C, H_p1, W_p1, H_p2, W_p2
        p = p.contiguous().view(B, C, -1, self.patch_size, self.patch_size) # B, C, H_p1, W_p1, H_p2, W_p2 -> B, C, N, p1, p2
        p = p.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, self.patch_size * self.patch_size * C) # B, C, N, p1, p2 -> B, N, C*p1*p2
        return p

class PatchEmbedding(nn.Module):
    def __init__(self, patch_elements, patch_num, emb_dim):
        super().__init__()
        self.patch_elements = patch_elements
        self.patch_num = patch_num
        self.proj_fc = spectral_norm(nn.Linear(patch_elements, emb_dim))
        self.position_embedding = nn.Embedding(patch_num, emb_dim)

    def forward(self, x):
        B = x.size()[0]

        positions = torch.arange(self.patch_num, device=x.device).unsqueeze(0).repeat(B, 1)
        #x = F.adaptive_avg_pool1d(, 1).squeeze(-1)
        p = self.proj_fc(x)
        q = self.position_embedding(positions) # [B, N, dim]
        encoded = p + q
        return encoded
    
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__()

        self.norm1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.attn = MultiHeadAttention_D(emb_dim=input_dim, output_dim=output_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.norm1(x)
        attn= self.attn(x1)
        x2 = attn + x
        x3 = self.norm1(x2)
        x3 = self.mlp(x3)
        out = x2 + x3
        return out


class MultiHeadAttention_D(nn.Module):
    def __init__(self, emb_dim, output_dim, num_heads, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** (0.5)

        self.qkv = spectral_norm(nn.Linear(emb_dim, emb_dim*3))
        self.attn_drop = nn.Dropout(dropout)
        self.proj = spectral_norm(nn.Linear(emb_dim, output_dim))

    def forward(self, x):
        B, N, C = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) # -> (3, B, heads, N, C//heads)
        q,k,v = qkv[0], qkv[1], qkv[2]

        energy = (q @ k.transpose(-2, -1)) / self.scale # -> (B, heads, N, C//heads) @ (B, heads, C//heads,  N) -> (B, heads, N, N)
        attn = F.softmax(energy, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2) # (B, heads, N, N) @ (B, heads, N, C//heads) -> (B, heads, N, C//heads) -> (B, N, heads, C//heads)
        out = out.reshape(B, N, C)
        out = self.proj(out)

        return out
