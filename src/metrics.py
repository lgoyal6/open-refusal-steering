"""Response rate, degeneracy, perplexity, latency.

The response-rate axis is the one everybody publishes. The other three are the
ones that decide whether a response-rate gain is real.
"""

from __future__ import annotations

import time

import torch

try:
    from .text_metrics import REFUSAL_MARKERS, PREFIX_CHARS, is_refusal, is_degenerate, classify
except ImportError:  # Existing scripts import metrics with src on sys.path.
    from text_metrics import REFUSAL_MARKERS, PREFIX_CHARS, is_refusal, is_degenerate, classify


@torch.no_grad()
def perplexity(model, tok, text: str, device: str, window: int = 512, stride: int = 512) -> float:
    """Token-level perplexity over a held-out corpus, non-overlapping windows.

    Called inside a steering context so the number reflects the capability cost
    of the intervention that is actually running.
    """
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    total_nll, total_tokens = 0.0, 0
    for start in range(0, ids.shape[1] - 1, stride):
        chunk = ids[:, start:start + window]
        if chunk.shape[1] < 2:
            break
        out = model(chunk, labels=chunk)
        n = chunk.shape[1] - 1  # number of predicted positions
        total_nll += out.loss.item() * n
        total_tokens += n
    return float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1))))


@torch.no_grad()
def time_per_token(model, tok, prompt: str, device: str, new_tokens: int = 48, reps: int = 3) -> float:
    """Median wall-clock milliseconds per generated token."""
    from steering import chat_prompt

    enc = tok(chat_prompt(tok, prompt), return_tensors="pt").to(device)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        dt = time.perf_counter() - t0
        generated = out.shape[1] - enc["input_ids"].shape[1]
        samples.append(dt * 1000.0 / max(generated, 1))
    samples.sort()
    return samples[len(samples) // 2]
