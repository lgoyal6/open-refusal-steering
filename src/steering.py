"""Activation steering by directional ablation on the residual stream.

Implements the intervention published by CTGT in "A feature-level approach to
mitigating bias and censorship in DeepSeek-R1" (hal-04992348v1):

    h' = h - alpha * (h . v_censor) * v_censor

where h is the hidden activation and alpha is a tunable scalar controlling
intervention strength. The paper gives no code; this file is that code.

Prior art: the same difference-in-means directional-ablation construction is
described in Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction" (arXiv:2406.11717). We reimplement it here rather than depend on it
so that the whole pipeline is one readable file.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


def chat_prompt(tok, user_msg: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def last_token_states(model, tok, prompts: list[str], device: str) -> torch.Tensor:
    """Hidden state at the final prompt token, for every layer.

    Returns a tensor of shape (n_prompts, n_layers + 1, d_model).
    """
    rows = []
    for p in prompts:
        enc = tok(chat_prompt(tok, p), return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        # hidden_states is a tuple of (n_layers + 1) tensors, each (1, seq, d)
        rows.append(torch.stack([h[0, -1, :].float() for h in out.hidden_states]))
    return torch.stack(rows)


def learn_direction(refuse_states: torch.Tensor, comply_states: torch.Tensor):
    """Difference-in-means direction per layer, plus a separation score per layer.

    refuse_states / comply_states: (n, n_layers + 1, d)
    Returns (directions, scores) where directions is (n_layers + 1, d), unit norm,
    and scores is (n_layers + 1,) holding Cohen's d of the projection onto the
    direction. The layer with the highest score is the one where a single linear
    direction best separates the two behaviours.
    """
    mu_r = refuse_states.mean(dim=0)
    mu_c = comply_states.mean(dim=0)
    diff = mu_r - mu_c                                    # (L+1, d)
    directions = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Cohen's d of the 1-D projection, per layer.
    proj_r = torch.einsum("nld,ld->nl", refuse_states, directions)
    proj_c = torch.einsum("nld,ld->nl", comply_states, directions)
    nr, nc = proj_r.shape[0], proj_c.shape[0]
    pooled = (
        ((nr - 1) * proj_r.var(dim=0) + (nc - 1) * proj_c.var(dim=0)) / (nr + nc - 2)
    ).sqrt().clamp_min(1e-8)
    scores = (proj_r.mean(dim=0) - proj_c.mean(dim=0)).abs() / pooled
    return directions, scores


class DirectionalAblation:
    """Applies h' = h - alpha (h . v) v to every decoder layer's residual stream.

    A single direction is applied at every layer and every token position, which
    is what a live inference-time intervention has to do. alpha = 0 is a no-op,
    alpha = 1 fully removes the component along v, alpha > 1 overshoots into the
    opposite half-space.
    """

    def __init__(self, model, direction: torch.Tensor, alpha: float):
        self.layers = model.model.layers
        self.v = direction / direction.norm().clamp_min(1e-8)
        self.alpha = float(alpha)
        self.handles = []

    def _hook(self, _module, _args, output):
        if self.alpha == 0.0:
            return output
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        v = self.v.to(device=h.device, dtype=h.dtype)
        # (h . v) v, broadcast over batch and sequence
        h = h - self.alpha * (h @ v).unsqueeze(-1) * v
        return (h,) + output[1:] if is_tuple else h

    def __enter__(self):
        self.handles = [ly.register_forward_hook(self._hook) for ly in self.layers]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False


@torch.no_grad()
def generate(model, tok, prompt: str, device: str, max_new_tokens: int = 96) -> str:
    enc = tok(chat_prompt(tok, prompt), return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
