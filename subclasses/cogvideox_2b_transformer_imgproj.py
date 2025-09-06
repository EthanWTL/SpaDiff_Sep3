
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel, CogVideoXBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogVideoXBlockWithImg(CogVideoXBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        # Build the original modules (norm1/attn1, norm2/ff) without duplicating logic
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            qk_norm=qk_norm,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            final_dropout=final_dropout,
            ff_inner_dim=ff_inner_dim,
            ff_bias=ff_bias,
            attention_out_bias=attention_out_bias,
        )

        # Add only your extras
        self.norm1_img = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn2 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        img_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ---- this part is identical to the parent up to before the FFN ----
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # ---- your injection point: second cross-attn to image tokens ----
        if img_hidden_states is not None:
            norm_hidden_states2, norm_img_hidden_states, gate_img, _ = self.norm1_img(
                hidden_states, img_hidden_states, temb
            )
            attn_hidden_states2, _ = self.attn2(
                hidden_states=norm_hidden_states2,
                encoder_hidden_states=norm_img_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            hidden_states = hidden_states + gate_img * attn_hidden_states2

        # ---- resume parent logic (norm2 + FFN) unchanged ----
        norm_hidden_states3, norm_encoder_hidden_states3, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
        norm_hidden_states3 = torch.cat([norm_encoder_hidden_states3, norm_hidden_states3], dim=1)
        ff_output = self.ff(norm_hidden_states3)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states   

class CogVideoXTransformer3DWithImgProj(CogVideoXTransformer3DModel):
    _no_split_modules = ["CogVideoXBlockWithImg", "CogVideoXPatchEmbed"]
    def __init__(self, *args, use_img_proj: bool = True, img_in_ch: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_to_config(use_img_proj=use_img_proj, img_in_ch=img_in_ch)

        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        # 1) optionally attach an image projector to patch_embed
        if self.config.use_img_proj and not hasattr(self.patch_embed, "img_proj"):
            if p_t is None:
                # 2D: project C -> inner_dim with conv kernel=stride=patch_size
                self.patch_embed.add_module(
                    "img_proj",
                    nn.Conv2d(
                        in_channels=self.config.img_in_ch,
                        out_channels=inner_dim,
                        kernel_size=p,
                        stride=p,
                        bias=True,
                    ),
                )
            else:
                # 3D: flat patch projection with Linear
                in_dim = self.config.img_in_ch * (p * p * p_t)
                self.patch_embed.add_module(
                    "img_proj",
                    nn.Linear(in_dim, inner_dim, bias=True),
                )

        # 2) rebuild transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CogVideoXBlockWithImg(
                dim=inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                time_embed_dim=self.config.time_embed_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                attention_bias=self.config.attention_bias,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
            )
            for _ in range(self.config.num_layers)
        ])
    
    #helper function to encode the condition image
    def _embed_condition_tokens(self, img_hidden_states: torch.Tensor, *, device, dtype) -> torch.Tensor:
        """
        Turn condition frames [B, F, C, H, W] into patch tokens [B, N_patches, inner_dim].
        If `patch_embed.img_proj` exists, we use it; otherwise we reuse patch_embed.forward
        with a dummy text block and slice out the image patch tokens.
        """
        B, F, C, H, W = img_hidden_states.shape
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        D = self.config.num_attention_heads * self.config.attention_head_dim

        # Preferred path: explicit projector (decoupled weights)
        if hasattr(self.patch_embed, "img_proj"):
            if p_t is None:
                # [B*F, C, H, W] -> [B, F, D, H/p, W/p] -> [B, F*H'*W', D]
                x = img_hidden_states.reshape(-1, C, H, W)
                x = self.patch_embed.img_proj(x)  # [B*F, D, H/p, W/p]
                Hp, Wp = H // p, W // p
                x = x.view(B, F, D, Hp, Wp).flatten(3).transpose(2, 3).flatten(1, 2)  # [B, F*Hp*Wp, D]
            else:
                # 3D: unfold (F x H x W) into patches and linearly project
                # [B, F, C, H, W] -> [B, F', H', W', (C*p*p*p_t)] -> Linear -> [B, N_patches, D]
                Fp = (F + p_t - 1) // p_t
                Hp, Wp = H // p, W // p
                x = img_hidden_states.permute(0, 1, 3, 4, 2)  # [B, F, H, W, C]
                x = x.reshape(B, Fp, p_t, Hp, p, Wp, p, C)
                x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)  # [B, Fp*Hp*Wp, C*p*p*p_t]
                x = self.patch_embed.img_proj(x)  # [B, N_patches, D]

            # add the same patch positional embeddings as the main stream
            pre_t = (F - 1) * self.config.temporal_compression_ratio + 1
            pos = self.patch_embed._get_positional_embeddings(H, W, pre_t, device=device)
            pos = pos[:, self.config.max_text_seq_length:, :].to(dtype=x.dtype)
            return x + pos

        # Fallback: reuse patch_embed.forward with dummy text, then slice the patch tokens
        dummy_txt = torch.zeros(
            B, self.config.max_text_seq_length, self.config.text_embed_dim, device=device, dtype=dtype
        )
        full = self.patch_embed(dummy_txt, img_hidden_states)  # [B, max_text+patches, D]
        return full[:, self.config.max_text_seq_length:, :]

    #forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        img_hidden_states: Optional[torch.Tensor] = None,  
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        # LoRA scaling plumbing (unchanged)
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` in attention_kwargs without PEFT backend has no effect.")

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1) time (and ofs) embeddings
        t_emb = self.time_proj(timestep).to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.ofs_proj(ofs).to(dtype=hidden_states.dtype)
            emb = emb + self.ofs_embedding(ofs_emb)

        # 2) patch-embed the main stream (text+video) as in base
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 2b) patch-embed the conditioning frames into tokens (if provided)
        img_tokens = None
        if img_hidden_states is not None:
            img_tokens = self._embed_condition_tokens(
                img_hidden_states, device=hidden_states.device, dtype=hidden_states.dtype
            )  # [B, N_cond, D]

        # 3) transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                    img_tokens,  
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                    img_tokens,  
                )

        hidden_states = self.norm_final(hidden_states)

        # 4) final + projection (unchanged)
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5) unpatchify (unchanged)
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)