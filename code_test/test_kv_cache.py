"""KV-cache correctness tests for the dense GPT.

Run from the project root:
    .venv/bin/python code_test/test_kv_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import GPT  # noqa: E402

CFG = {
    "vocab_size": 64,
    "context_length": 16,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "tie_weights": True,
}

ATOL = 1e-5
RTOL = 1e-4


def make_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    model = GPT(CFG)
    model.eval()
    return model


def assert_close(actual, expected, label: str) -> None:
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL, msg=label)


def test_training_path_untouched() -> None:
    model = make_model()
    x = torch.randint(0, CFG["vocab_size"], (2, 10))
    y = torch.randint(0, CFG["vocab_size"], (2, 10))

    model.train()
    logits, loss = model(x, y)
    assert logits.shape == (2, 10, CFG["vocab_size"])
    assert loss.ndim == 0
    loss.backward()
    assert model.kv_caches is None
    assert not any("kv_cache" in key for key in model.state_dict())

    model.eval()
    model(x, use_cache=True)
    before = model.kv_caches[0].seq_len
    model(x)
    assert model.kv_caches[0].seq_len == before

    try:
        model(x, y=y, use_cache=True)
    except ValueError:
        pass
    else:
        raise AssertionError("use_cache=True with targets must raise")
    print("training path untouched OK")


def test_prefill_chunk_and_token_decode() -> None:
    model = make_model()
    x = torch.randint(0, CFG["vocab_size"], (2, 12))
    ref = model(x)

    model.reset_cache()
    got = model(x, use_cache=True)
    assert_close(got, ref, "cached prefill")
    assert model.kv_caches[0].seq_len == x.size(1)

    model.reset_cache()
    _ = model(x[:, :5], use_cache=True)
    chunk = model(x[:, 5:], use_cache=True)
    assert_close(chunk, ref[:, 5:], "chunked decode")

    model.reset_cache()
    pieces = []
    for i in range(x.size(1)):
        pieces.append(model(x[:, i : i + 1], use_cache=True))
    token_by_token = torch.cat(pieces, dim=1)
    assert_close(token_by_token, ref, "token-by-token decode")
    print("prefill/chunk/token decode OK")


def test_cache_contains_positioned_normalized_k() -> None:
    model = make_model()
    x = torch.randint(0, CFG["vocab_size"], (1, 9))
    h = model.drop_emb(
        model.tok_emb(x)
        + model.pos_emb(torch.arange(x.size(1), device=x.device))
    )
    block = model.trf_blocks[0]
    expected_k = block.att.W_key(block.norm1(h)).view(
        1, x.size(1), CFG["n_heads"], block.att.head_dim
    ).transpose(1, 2)

    model.reset_cache()
    model(x, use_cache=True)
    assert_close(
        model.kv_caches[0].k_cache[:, :, : x.size(1)],
        expected_k,
        "cached K values",
    )
    print("cached K values OK")


def test_generate_cached_matches_uncached() -> None:
    model = make_model(seed=7)
    prompt = torch.randint(0, CFG["vocab_size"], (2, 5))
    model.train()

    torch.manual_seed(11)
    cached = model.generate(
        prompt, max_new_tokens=6, temperature=0.0, use_cache=True
    )
    torch.manual_seed(11)
    plain = model.generate(
        prompt, max_new_tokens=6, temperature=0.0, use_cache=False
    )
    assert torch.equal(cached, plain)
    assert torch.equal(cached[:, : prompt.size(1)], prompt)
    assert model.training is True

    torch.manual_seed(12)
    cached_topk = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        top_k=5,
        use_cache=True,
    )
    torch.manual_seed(12)
    plain_topk = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        top_k=5,
        use_cache=False,
    )
    assert torch.equal(cached_topk, plain_topk)

    model.eval()
    model.generate(prompt, max_new_tokens=2, temperature=0.0, use_cache=True)
    assert model.training is False
    print("cached generate matches uncached OK")


def test_batch_resize_and_reset() -> None:
    model = make_model(seed=3)
    prompt2 = torch.randint(0, CFG["vocab_size"], (2, 4))
    prompt1 = prompt2[:1]

    out2 = model.generate(
        prompt2, max_new_tokens=3, temperature=0.0, use_cache=True
    )
    assert out2.shape == (2, 7)
    out1 = model.generate(
        prompt1, max_new_tokens=3, temperature=0.0, use_cache=True
    )
    assert out1.shape == (1, 7)
    assert model.kv_caches[0].batch_size == 1

    torch.manual_seed(5)
    out1_plain = model.generate(
        prompt1, max_new_tokens=3, temperature=0.0, use_cache=False
    )
    assert torch.equal(out1, out1_plain)
    print("batch resize/reset OK")


def test_context_window() -> None:
    model = make_model(seed=4)
    full_prompt = torch.randint(0, CFG["vocab_size"], (1, CFG["context_length"]))

    out = model.generate(
        full_prompt, max_new_tokens=1, temperature=0.0, use_cache=True
    )
    assert out.shape == (1, CFG["context_length"] + 1)

    cached = model.generate(
        full_prompt, max_new_tokens=2, temperature=0.0, use_cache=True
    )
    plain = model.generate(
        full_prompt, max_new_tokens=2, temperature=0.0, use_cache=False
    )
    assert torch.equal(cached, plain)
    assert cached.shape == (1, CFG["context_length"] + 2)

    long_prompt = torch.randint(
        0, CFG["vocab_size"], (1, CFG["context_length"] + 4)
    )
    cached = model.generate(
        long_prompt, max_new_tokens=1, temperature=0.0, use_cache=True
    )
    plain = model.generate(
        long_prompt, max_new_tokens=1, temperature=0.0, use_cache=False
    )
    assert torch.equal(cached, plain)
    assert cached.shape == (1, long_prompt.size(1) + 1)

    cached_long = model.generate(
        long_prompt,
        max_new_tokens=CFG["context_length"] + 2,
        temperature=0.0,
        use_cache=True,
    )
    plain_long = model.generate(
        long_prompt,
        max_new_tokens=CFG["context_length"] + 2,
        temperature=0.0,
        use_cache=False,
    )
    assert torch.equal(cached_long, plain_long)
    assert cached_long.size(1) == long_prompt.size(1) + CFG["context_length"] + 2
    print("context window behavior OK")


def test_eos_padding_and_sampling_validation() -> None:
    model = make_model()
    eos = 3
    finished = torch.tensor([False, False])
    idx_next = torch.tensor([[eos], [5]])
    idx_next, finished, all_done = model._apply_eos(idx_next, finished, eos)
    assert torch.equal(idx_next, torch.tensor([[eos], [5]]))
    assert torch.equal(finished, torch.tensor([True, False]))
    assert all_done is False

    idx_next = torch.tensor([[7], [eos]])
    idx_next, finished, all_done = model._apply_eos(idx_next, finished, eos)
    assert torch.equal(idx_next, torch.tensor([[eos], [eos]]))
    assert torch.equal(finished, torch.tensor([True, True]))
    assert all_done is True

    prompt = torch.randint(0, CFG["vocab_size"], (2, 4))
    out = model.generate(
        prompt, max_new_tokens=4, temperature=0.8, top_k=5, eos_id=None
    )
    assert out.shape == (2, 8)
    assert int(out.min()) >= 0 and int(out.max()) < CFG["vocab_size"]

    for kwargs in ({"top_k": 0}, {"temperature": -0.1}):
        try:
            model.generate(prompt, max_new_tokens=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid kwargs did not raise: {kwargs}")
    print("EOS padding / sampling validation OK")


def main() -> None:
    test_training_path_untouched()
    test_prefill_chunk_and_token_decode()
    test_cache_contains_positioned_normalized_k()
    test_generate_cached_matches_uncached()
    test_batch_resize_and_reset()
    test_context_window()
    test_eos_padding_and_sampling_validation()
    print("all KV-cache tests passed")


if __name__ == "__main__":
    main()
