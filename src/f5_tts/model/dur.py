import einx
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, Tensor
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.backbones.dit import TextEmbedding
from f5_tts.model.modules import MelSpec, Attention, ConvPositionEmbedding, FeedForward
from f5_tts.model.utils import (
    exists,
    default,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    maybe_masked_mean,
)

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
SAMPLES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH


# reference: https://github.com/lucasnewman/f5-tts-mlx/blob/4d24ebcc0c7c6215d64ed29eabdd32570084543f/f5_tts_mlx/duration.py
class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: Tensor) -> Tensor:
        return rearrange(x, self.pattern)


class DurationInputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim, out_dim)
        # ConvPositionEmbedding是一个利用1D卷积引入位置信息的模块。这是一种比固定sin/cos位置编码更灵活的方法！
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
            self,
            x: Tensor,  # shape: [batch_size, seq_len, mel_dim]
            text_embed: Tensor,  # shape: [batch_size, seq_len, text_dim]
    ) -> Tensor:  # shape: [batch_size, seq_len, out_dim]
        x = self.proj(torch.cat((x, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DurationBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh"
        )

    def forward(self, x, mask=None, rope=None):
        norm = self.attn_norm(x)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + attn_output

        norm = self.ff_norm(x)
        ff_output = self.ff(norm)
        x = x + ff_output

        return x


class DurationTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth=8,
            heads=8,
            dim_head=64,
            dropout=0.0,
            ff_mult=4,
            mel_dim=100,
            text_num_embeds=256,
            text_dim=None,
            conv_layers=0,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, conv_layers=conv_layers
        )
        self.input_embed = DurationInputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = [
            DurationBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ]

        self.norm_out = nn.RMSNorm(dim)

    def forward(
            self,
            x: torch.Tensor,  # noisy input audio (batch_size, seq_len, mel_dim)
            text: torch.Tensor,  # text (batch_size, text_seq_len)
            mask: torch.Tensor | None = None,  # (batch_size, seq_len)
    ):
        seq_len = x.shape[1]

        text_embed = self.text_embed(text, seq_len)

        x = self.input_embed(x, text_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, mask=mask, rope=rope)

        x = self.norm_out(x)

        return x


class DurationPredictor(nn.Module):
    def __init__(
            self,
            transformer: DurationTransformer,
            num_channels=None,
            mel_spec_kwargs: dict = dict(),
            vocab_char_map: dict[str, int] | None = None,
    ):
        super().__init__()

        # mel spec
        self._mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = default(num_channels, self._mel_spec.n_mels)
        self.num_channels = num_channels

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # vocab map for tokenization
        self._vocab_char_map = vocab_char_map

        # to prediction

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1, bias=False), nn.Softplus(), Rearrange("... 1 -> ...")
        )

    def forward(
            self,
            inp: float["b n d"] | float["b nw"],  # noqa: F722
            text: int["b nt"] | list[str],  # noqa: F722
            *,
            lens: int["b"] | None = None,  # noqa: F821
            return_loss=False,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self._mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len = inp.shape[:2]
        device = inp.device

        # handle text as string
        if isinstance(text, list):
            if exists(self._vocab_char_map):
                text = list_str_to_idx(text, self._vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if seq_len < text.shape[1]:
            seq_len = text.shape[1]
            # 第一维b不填充，剩下的维度是2d张量，形状为(left, right, top, bottom)，填充的是行的底部
            inp = F.pad(inp, (0, 0, 0, seq_len - inp.shape[1]))

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        # lens是实际的mel长度，seq_len是整个batch的对齐长度。下面操作lens是tensor([1,2,3])，seq_len是4
        # tensor([[ True, False, False, False],
        #         [ True,  True, False, False],
        #         [ True,  True,  True, False]])
        mask = lens_to_mask(lens, length=seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration
        # 随机从某个索引位置盖住，比如
        # before tensor([[True, True, False],
        #                [True, True, True]])
        # rand_index tensor([1, 1])，都从索引1开始掩码
        # after tensor([[True, False, False],
        #               [True, False, False]])
        if return_loss:
            # 生成[0,1)之间的随机数
            rand_frac_index = torch.rand(batch, device=device)
            # 将随机数乘以序列长度并转为整数
            rand_index = (rand_frac_index * lens).to(torch.int32)
            # 创建序列索引
            seq = torch.arange(seq_len, device=device)
            mask &= einx.less("n, b -> b n", seq, rand_index)

        # attending
        inp = torch.where(
            mask.unsqueeze(-1).expand(-1, -1, self.num_channels),
            inp,
            torch.zeros_like(inp)
        )
        x = self.transformer(inp, text=text)

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        # return the prediction if not returning loss
        if not return_loss:
            return pred

        # loss
        duration = lens.to(pred.dtype) / SAMPLES_PER_SECOND

        return F.l1_loss(pred, duration)