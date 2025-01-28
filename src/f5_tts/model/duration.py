import einx
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from f5_tts.model.modules import MelSpec, AttnProcessor, Attention, ConvPositionEmbedding, FeedForward
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

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq
        # self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.scale = None
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.scale = scale

    def forward_from_seq_len(self, seq_len: int) -> tuple[torch.Tensor, float]:
        t = torch.arange(seq_len)
        return self(t)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, float]:
        max_pos = t.max() + 1

        freqs = torch.einsum("i , j -> i j", t.to(self.inv_freq.dtype), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), axis=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if self.scale is None:
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.stack((scale, scale), axis=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale


class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: torch.array) -> torch.array:
        return rearrange(x, self.pattern)


class DurationInputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def __call__(
        self,
        x: float["b n d"],  # noqa: F722
        text_embed: float["b n d"],  # noqa: F722
    ):
        x = self.proj(torch.concatenate((x, text_embed), axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Duration block


class DurationBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def __call__(self, x, mask=None, rope=None):
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

        # Causes circular import if global, maybe could move duration to it's own function but works ok like this
        from f5_tts.model.backbones.dit import TextEmbedding

        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_embed = DurationInputEmbedding(mel_dim, text_dim, dim).to(self.device)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DurationBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = nn.RMSNorm(dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio # noqa: F722
        text: int["b nt"],  # text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        seq_len = x.shape[1]
        text_embed = self.text_embed(text, seq_len).to(self.device)

        x = self.input_embed(x, text_embed).to("cpu")

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = default(num_channels, self._mel_spec.n_mel_channels)
        self.num_channels = num_channels

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        self.dim = dim

        # vocab map for tokenization
        self._vocab_char_map = vocab_char_map

        # to prediction

        self.to_pred = nn.Sequential(nn.Linear(dim, 1, bias=False), nn.Softplus(), Rearrange("... 1 -> ..."))

    def forward(
        self,
        inp: torch.Tensor["b n d"] | torch.Tensor["b nw"],  # mel or raw wave # noqa: F722
        text: torch.Tensor | list[str],
        *,
        lens: torch.Tensor["b"] | None = None,  # noqa: F821
        return_loss=False,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self._mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len = inp.shape[:2]

        # handle text as string
        if isinstance(text, list):
            if exists(self._vocab_char_map):
                text = list_str_to_idx(text, self._vocab_char_map)
            else:
                text = list_str_to_tensor(text)
            assert text.shape[0] == batch

        if seq_len < text.shape[1]:
            seq_len = text.shape[1]
            inp = F.pad(inp, [(0, 0), (0, seq_len - inp.shape[1]), (0, 0)])

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len)

        if seq_len < text.shape[1]:
            seq_len = text.shape[1]
            inp = F.pad(inp, [(0, 0), (0, seq_len - inp.shape[1]), (0, 0)])

        mask = lens_to_mask(lens, length=seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = torch.random.uniform(0, 1, (batch,))
            rand_index = (rand_frac_index * lens).astype(torch.int32)

            seq = torch.arange(seq_len)
            mask &= einx.less("n, b -> b n", seq, rand_index)

        # attending

        mask.to(self.device)
        inp.to(self.device)

        inp = torch.where(repeat(mask, "b n -> b n d", d=self.num_channels).to(self.device), inp, torch.zeros_like(inp).to(self.device))

        x = self.transformer(inp, text=text)

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        # return the prediction if not returning loss

        if not return_loss:
            return pred

        # loss

        duration = lens.to(pred.dtype) / SAMPLES_PER_SECOND

        return nn.losses.l1_loss(pred, duration)