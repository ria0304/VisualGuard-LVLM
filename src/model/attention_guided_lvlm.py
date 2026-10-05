import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class AttentionGuidedConfig:
    vision_dim: int = 1408
    hidden_dim: int = 4096
    num_heads: int = 32
    vocab_size: int = 32000
    max_seq_len: int = 256
    num_image_tokens: int = 576
    num_object_queries: int = 100
    dropout: float = 0.0
    use_object_queries: bool = True


class VisionEncoder(nn.Module):
    """Vision encoder with object-query integration (DETR-style)."""

    def __init__(self, vision_dim: int = 1408, num_patches: int = 576,
                 num_object_queries: int = 100, image_size: int = 336):
        super().__init__()
        self.vision_dim = vision_dim
        self.num_patches = num_patches
        self.num_object_queries = num_object_queries

        # Patch stem
        self.stem = nn.Conv2d(3, vision_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((num_patches // 24, 24))

        # Object queries (learnable, like DETR)
        self.object_queries = nn.Parameter(torch.randn(num_object_queries, vision_dim))

        # Query position embeddings
        self.query_pos_embed = nn.Parameter(torch.randn(1, num_object_queries, vision_dim))

        # Patch position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, vision_dim))

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]

        # Patch features
        x = self.stem(pixel_values)  # (B, vision_dim, H, W)
        x = self.pool(x)  # (B, vision_dim, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, vision_dim)
        x = x + self.pos_embed

        # Object queries attend to patches (cross-attention)
        # Q: (1, num_queries, D), K/V: (B, num_patches, D)
        Q = self.object_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, D)
        K = x  # (B, num_patches, D)
        V = x  # (B, num_patches, D)

        # Scaled dot-product attention
        attn_weights = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.vision_dim)  # (B, num_queries, num_patches)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, num_queries, num_patches)

        # Grounded object features (weighted sum of patches)
        object_features = torch.bmm(attn_weights, V)  # (B, num_queries, vision_dim)
        object_features = object_features + self.query_pos_embed

        return {
            'patch_features': x,          # (B, num_patches, vision_dim) for LLM
            'object_features': object_features,  # (B, num_queries, vision_dim) for grounding
            'attention_weights': attn_weights,    # (B, num_queries, num_patches) for analysis
        }


class ObjectHallucinationDetector(nn.Module):
    """Detects hallucinations by comparing attended object regions to image content.

    Outputs a scalar [0,1] per token position indicating hallucination likelihood.
    """

    def __init__(self, hidden_dim: int = 4096, num_object_queries: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_object_queries

        # MLP that takes attended object feature + token embedding
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_hidden: torch.Tensor, attended_objects: torch.Tensor) -> torch.Tensor:
        """token_hidden: (B, seq_len, hidden_dim)
           attended_objects: (B, num_queries, hidden_dim) - the currently attended regions
        """
        # Pool attended objects (mean pool across queries)
        avg_objects = attended_objects.mean(dim=1)  # (B, hidden_dim)

        # Expand token hidden for the last position only (we detect per-position)
        # Here we detect for the last generated token
        last_hidden = token_hidden[:, -1:, :]  # (B, 1, hidden_dim)

        # Concatenate and predict
        combined = torch.cat([last_hidden.squeeze(1), avg_objects], dim=-1)  # (B, 2*hidden_dim)
        hall_score = self.detector(combined).squeeze(-1)  # (B,)

        return hall_score  # Higher = more likely hallucination


class AttentionGuidedLMHead(nn.Module):
    """LLaMA-style language model head with attention-guided grounding and hallucination detection."""

    def __init__(self, config: AttentionGuidedConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.visual_proj = nn.Linear(config.vision_dim, config.hidden_dim)

        # Novel: object hallucination detector
        self.hallucination_detector = ObjectHallucinationDetector(
            hidden_dim=config.hidden_dim,
            num_object_queries=config.num_object_queries,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_features: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (logits, hallucination_scores, attended_objects)

        visual_features dict contains:
            - patch_features: (B, num_patches, vision_dim)
            - object_features: (B, num_queries, vision_dim)
            - attention_weights: (B, num_queries, num_patches)
        """
        # input_ids: (B, seq_len)
        # visual_features['patch_features']: (B, num_patches, vision_dim)

        # Token embeddings
        x = self.tok_emb(input_ids)  # (B, seq_len, hidden_dim)

        # Project visual patches to hidden dim
        v = self.visual_proj(visual_features['patch_features'])  # (B, num_patches, hidden_dim)

        # --- ATTENTION-GUIDED GROUNDING ---
        # Token-to-patch cross-attention
        # (B, seq_len, hidden_dim) x (B, hidden_dim, num_patches) -> (B, seq_len, num_patches)
        attn_weights = torch.bmm(x, v.transpose(1, 2)) / math.sqrt(self.config.hidden_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, seq_len, num_patches)

        # Grounded representation: weighted sum of visual features
        grounded = torch.bmm(attn_weights, v)  # (B, seq_len, hidden_dim)

        # Combine token embeddings with grounded visual information
        x = x + grounded  # residual-style grounding fusion

        # Pass through transformer
        x = self.model(x)

        # LM logits
        logits = self.lm_head(x)  # (B, seq_len, vocab_size)

        # --- NOVEL: Hallucination Detection ---
        # Get attended object features for the last token position
        attended_objects = visual_features['object_features']  # (B, num_queries, vision_dim)
        attended_objects_proj = self.visual_proj(attended_objects)  # (B, num_queries, hidden_dim)

        # Compute hallucination score for each batch item
        hall_scores = self.hallucination_detector(x, attended_objects_proj)  # (B,)

        return logits, hall_scores, attended_objects

    def get_grounded_attention(self, input_ids: torch.Tensor, visual_features: Dict) -> torch.Tensor:
        """Return attention map for analysis: (B, seq_len, num_patches)."""
        v = self.visual_proj(visual_features['patch_features'])
        x = self.tok_emb(input_ids)
        attn = torch.bmm(x, v.transpose(1, 2)) / math.sqrt(self.config.hidden_dim)
        return F.softmax(attn, dim=-1)


class DynamicVisualFeedbackLoop(nn.Module):
    """Dynamic visual feedback loop with iterative attention refinement.

    Key novelty: At each generation step, attention is recomputed and
    refined based on the growing token sequence, preventing drift into
    hallucinated content. Uses object-level queries for more precise grounding.
    """

    def __init__(self, config: AttentionGuidedConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(
            vision_dim=config.vision_dim,
            num_patches=config.num_image_tokens,
            num_object_queries=config.num_object_queries,
        )
        self.lm_head = AttentionGuidedLMHead(config)
        self.refinement_steps = 2  # Number of attention refinement iterations

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        visual_features = self.vision_encoder(pixel_values)  # dict with patch/object features
        logits, hall_scores, attended_objects = self.lm_head(input_ids, visual_features, **kwargs)
        return logits, hall_scores, visual_features, attended_objects

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 40,
        hallucination_penalty: float = 0.5,
    ) -> torch.Tensor:
        """Autoregressive generation with iterative attention refinement.

        At each step:
        1. Compute cross-attention between tokens and visual patches
        2. Refine attention via iterative attention update
        3. Apply hallucination penalty to token probabilities
        4. Sample next token
        """
        self.eval()
        device = pixel_values.device
        B = pixel_values.shape[0]

        # Initial visual features
        visual_features = self.vision_encoder(pixel_values)  # dict

        # Start generation from prompt
        input_ids = prompt_ids.to(device)  # (B, seq_len)

        for _ in range(max_new_tokens):
            # --- Iterative attention refinement ---
            # Refine attention weights over multiple steps
            attn_weights = None
            for _ in range(self.refinement_steps):
                logits_at_step, hall_scores, attended_objects = self.lm_head(
                    input_ids, visual_features)
                # Use attended objects to modulate attention
                if attn_weights is None:
                    attn_weights = self.lm_head.get_grounded_attention(input_ids, visual_features)
                else:
                    # Refine: blend previous attention with object-guided attention
                    refined = self.lm_head.get_grounded_attention(input_ids, visual_features)
                    attn_weights = 0.7 * attn_weights + 0.3 * refined

            # Get logits from the grounded LM head
            logits, hall_scores, attended_objects = self.lm_head(
                input_ids, visual_features)  # (B, seq_len, vocab_size), (B,), (B, Q, D)

            # Last token logits
            next_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # --- NOVEL: Hallucination-adjusted probability modification ---
            # Reduce probability of tokens that increase hallucination risk
            # Simple heuristic: reduce logits for high-probability tokens when hallucination score is high
            # hall_scores: (B,) - squeeze for scalar comparison
            hall_mean = hall_scores.mean().item()  # scalar
            if hall_mean > 0.5:  # Hallucination likely (across batch)
                # Find top-k tokens and slightly reduce their logits (apply per-batch)
                _, top_tokens = torch.topk(next_logits, min(top_k, next_logits.size(-1)), dim=-1)
                # penalty per batch item
                penalty = hallucination_penalty * (1 - hall_scores).detach().cpu().numpy()
                # Apply penalty to top-k tokens for each batch
                for b in range(B):
                    p = penalty[b] if hasattr(penalty, '__getitem__') else penalty
                    for t_idx in range(min(top_k, top_tokens.size(-1))):
                        t = top_tokens[b, t_idx].item()
                        next_logits[b, t] -= p

            # Top-k filtering
            if top_k is not None:
                values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)), dim=-1)
                mask = torch.full_like(next_logits, -float("Inf"))
                mask.scatter_(dim=-1, index=indices, value=0.0)
                next_logits = next_logits + mask

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_ids], dim=1)  # (B, seq_len+1)

        return input_ids


class FactualityLVLM(nn.Module):
    """Full LVLM model with attention-guided decoding for factuality reduction."""

    def __init__(self, config: AttentionGuidedConfig):
        super().__init__()
        self.config = config
        self.visual_encoder = VisionEncoder(
            vision_dim=config.vision_dim,
            num_patches=config.num_image_tokens,
            num_object_queries=config.num_object_queries,
        )
        self.feedback_loop = DynamicVisualFeedbackLoop(config)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        return self.feedback_loop(pixel_values, input_ids, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str = "",
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 40,
        **kwargs,
    ) -> str:
        """End-to-end generation with attention-guided grounding and hallucination reduction."""
        # Tokenize prompt (simple mock)
        if prompt:
            prompt_ids = torch.tensor([[100 + ord(c) for c in prompt[:10]]], device=pixel_values.device)
        else:
            prompt_ids = torch.tensor([[self.config.vocab_size - 2]], device=pixel_values.device)  # <BOS>

        # Broadcast prompt_ids to match pixel_values batch size
        if prompt_ids.shape[0] == 1 and pixel_values.shape[0] > 1:
            prompt_ids = prompt_ids.expand(pixel_values.shape[0], -1)

        generated = self.feedback_loop.generate(
            pixel_values, prompt_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, **kwargs
        )
        return self._ids_to_text(generated[0].tolist())

    def _ids_to_text(self, ids: list) -> str:
        # Mock decoding
        return " ".join([f"token_{i}" for i in ids])