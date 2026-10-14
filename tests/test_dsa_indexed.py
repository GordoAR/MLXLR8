import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exlr8 import dsa_indexed


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host")


def fake_inputs(length):
    shapes = [(1, 64, length, 512), (1, 64, length, 64),
              (1, 1, 4096, 512), (1, 1, 4096, 64)]
    values = [SimpleNamespace(shape=shape, dtype=mx.bfloat16) for shape in shapes]
    return values + [SimpleNamespace(shape=(1, 1, length, 2048), dtype=mx.uint32), None]


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 128, 512, 513])
def test_installer_routes_only_supported_queries(monkeypatch, length):
    args, calls = fake_inputs(length), []
    native = lambda *values: calls.append(("native", values)) or "native"
    monkeypatch.setattr(dsa_indexed, "_indexed_attention", lambda *values: calls.append(("indexed", values)) or "indexed")
    monkeypatch.setattr(dsa_indexed, "_indexed_decode", lambda *values: calls.append(("decode", values)) or "decode")
    attention = SimpleNamespace(scale=0.0625, _sparse_attention=native)
    dsa_indexed.install_indexed_prefill(attention)
    expected = "decode" if length <= 4 else "indexed" if length >= 32 else "native"
    assert attention._sparse_attention(*args) == expected
    assert len(calls) == 1 and calls[0][0] == expected
    assert calls[0][1][:6] == tuple(args)
    if expected == "indexed":
        assert calls[0][1][6:] == (attention.scale, native)
    elif expected == "decode":
        assert calls[0][1][6:] == (attention.scale,)


@pytest.mark.parametrize("change", ["batch", "heads", "dims", "pe", "keys", "selected", "indices", "signed_ids", "rank", "mask", 0, 1, 2, 3])
@pytest.mark.parametrize("length", [1, 4, 64])
def test_installer_keeps_unsupported_inputs_native(monkeypatch, change, length):
    args, calls = fake_inputs(length), []
    if isinstance(change, int):
        args[change].dtype = mx.float32
    elif change == "batch":
        args[0].shape = (2, 64, length, 512)
    elif change == "heads":
        args[0].shape = (1, 32, length, 512)
    elif change == "dims":
        args[0].shape = (1, 64, length, 256)
    elif change == "pe":
        args[1].shape = (1, 64, length, 32)
    elif change == "keys":
        args[2].shape = (1, 1, 2047, 512)
    elif change == "selected":
        args[4].shape = (1, 1, length, 1024)
    elif change == "indices":
        args[4] = None
    elif change == "signed_ids":
        args[4].dtype = mx.int32
    elif change == "rank":
        args[0].shape = (64, length, 512)
    elif change == "mask":
        args[5] = SimpleNamespace(dtype=mx.float32)
    native = lambda *values: calls.append(values) or "native"
    monkeypatch.setattr(dsa_indexed, "_indexed_attention", lambda *values: pytest.fail("unsupported input reached Metal"))
    monkeypatch.setattr(dsa_indexed, "_indexed_decode", lambda *values: pytest.fail("unsupported input reached Metal"))
    attention = SimpleNamespace(scale=0.0625, _sparse_attention=native)
    dsa_indexed.install_indexed_prefill(attention)
    assert attention._sparse_attention(*args) == "native"
    assert calls == [tuple(args)]


def random_inputs(length):
    rng = np.random.default_rng(659 + length)
    keys = 2304 + length
    def random(shape):
        return mx.array(rng.normal(0, 0.4, shape).astype(np.float32), mx.bfloat16)
    ids = np.stack([np.concatenate(([0], rng.choice(np.arange(1, keys), 2047, replace=False)))
                    for _ in range(length)]).astype(np.uint32)
    return (random((1, 64, length, 512)), random((1, 64, length, 64)),
            random((1, 1, keys, 512)), random((1, 1, keys, 64)), mx.array(ids[None, None]))


def assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype == mx.bfloat16
    a, b = [np.asarray(value.astype(mx.float32)) for value in (actual, expected)]
    assert np.all(np.isfinite(a))
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


@metal
def test_indexed_products_keep_native_bf16_boundaries():
    q, _, latent, _, ids = random_inputs(32)
    q = q.swapaxes(1, 2).reshape(32, 64, 512) * 0.0625
    ids, latent = ids.reshape(32, 2048), latent.reshape(-1, 512)
    values = mx.take_along_axis(mx.broadcast_to(latent[None], (32, latent.shape[0], 512)),
        mx.broadcast_to(ids[..., None], (32, 2048, 512)), axis=1)
    score = q @ values.swapaxes(-1, -2)
    assert_bits_equal(dsa_indexed._qk(q, latent, ids), score)
    probabilities = mx.softmax(score, axis=-1, precise=True)
    assert_bits_equal(dsa_indexed._pv(probabilities, latent, ids), probabilities @ values)


@metal
@pytest.mark.parametrize("length,mask_rank", [(32, None), (33, 1), (65, 2), (545, 4)])
def test_prefill_keeps_ordered_ids_masks_and_native_row_tails(length, mask_rank):
    from exlr8.model import Attention
    args = list(random_inputs(length))
    keys = args[2].shape[2]
    mask = None
    if mask_rank is not None:
        mask = mx.arange(keys) % 5 != 4
        if mask_rank == 2:
            mask = mx.broadcast_to(mask, (length, keys)) & ((mx.arange(length)[:, None] + mx.arange(keys)[None]) % 7 != 6)
        elif mask_rank == 4:
            mask = mask[None, None, None]
    args.append(mask)
    attention = SimpleNamespace(scale=0.0625)
    attention._sparse_attention = lambda *values: Attention._sparse_attention(attention, *values)
    expected = attention._sparse_attention(*args)
    dsa_indexed.install_indexed_prefill(attention)
    actual = attention._sparse_attention(*args)
    assert actual.shape == (1, 64, length, 512)
    assert_bits_equal(actual, expected)


@metal
@pytest.mark.parametrize("length", [128, 512])
def test_actual_hermes_prefill_keeps_native_bits(length):
    from exlr8.model import Attention
    from exlr8.weights import SafeTensorFile
    path = Path(__file__).resolve().parents[1] / "artifacts/dsa-indexed-attention-inputs.safetensors"
    if not path.exists():
        pytest.skip("actual Hermes attention fixture is not present")
    file = SafeTensorFile(path)
    q, qp, latent, pe, ids = [file.load("prefill." + name) for name in ("q", "q_pe", "latent", "pe", "ids")]
    args = (q[-length:].swapaxes(0, 1)[None], qp[-length:].swapaxes(0, 1)[None],
            latent[None, None], pe[None, None], ids[-length:][None, None], None)
    attention = SimpleNamespace(scale=float(file.metadata["prefill.scale"]))
    attention._sparse_attention = lambda *values: Attention._sparse_attention(attention, *values)
    expected = attention._sparse_attention(*args)
    dsa_indexed.install_indexed_prefill(attention)
    assert_bits_equal(attention._sparse_attention(*args), expected)


@metal
def test_decode_products_keep_masked_bf16_boundaries_and_native_partitions():
    q, qp, latent, pe, ids = random_inputs(1)
    q, qp = q.reshape(1, 64, 512) * 0.0625, qp.reshape(1, 64, 1, 64) * 0.0625
    ids, latent, pe = ids.reshape(1, 2048), latent.reshape(-1, 512), pe.reshape(-1, 64)
    values = mx.take_along_axis(latent[None], ids[..., None], axis=1)
    pe_values = mx.take_along_axis(pe[None], ids[..., None], axis=1).reshape(1, 1, 2048, 64)
    valid = ids % 7 != 6
    score = (q @ values.swapaxes(-1, -2)) + mx.where(valid[:, None],
        (qp @ pe_values.swapaxes(-1, -2)).reshape(1, 64, 2048), -float("inf"))
    actual = dsa_indexed._decode_scores(q, qp.reshape(1, 64, 64), latent, pe, ids, valid)
    assert actual.dtype == score.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)), np.asarray(score.view(mx.uint16)))
    p, expected_p = mx.softmax(actual, axis=-1, precise=True), mx.softmax(score, axis=-1, precise=True)
    assert_bits_equal(p, expected_p)
    assert_bits_equal(dsa_indexed._decode_pv(p, latent, ids), expected_p @ values)


@metal
@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("masked", [False, True])
def test_decode_batch_matches_independent_queries_with_causal_masks(length, masked):
    q, qp, latent, pe, ids = random_inputs(length)
    keys = latent.shape[2]
    mask = None
    if masked:
        mask = ((mx.arange(length)[:, None] + mx.arange(keys)[None]) % 7 != 6)
    actual = dsa_indexed._indexed_decode(q, qp, latent, pe, ids, mask, 0.0625)
    expected = []
    for row in range(length):
        row_mask = mx.arange(keys) <= keys - length + row
        if mask is not None:
            row_mask &= mask[row]
        expected.append(dsa_indexed._indexed_decode(q[:, :, row:row + 1], qp[:, :, row:row + 1],
            latent, pe, ids[:, :, row:row + 1], row_mask, 0.0625))
    assert actual.shape == (1, 64, length, 512)
    assert_bits_equal(actual, mx.concatenate(expected, axis=2))


@metal
@pytest.mark.parametrize("length", [2, 3, 4])
def test_decode_batch_cannot_read_future_values(length):
    q, qp, latent, pe, ids = random_inputs(length)
    keys = latent.shape[2]
    ids = mx.concatenate([mx.broadcast_to(mx.arange(keys - length, keys, dtype=mx.uint32),
        (1, 1, length, length)), ids[..., length:]], axis=-1)
    actual = dsa_indexed._indexed_decode(q, qp, latent, pe, ids, None, 0.0625)
    changed_latent = mx.concatenate([latent[:, :, :-1], latent[:, :, -1:] + 5], axis=2)
    changed_pe = mx.concatenate([pe[:, :, :-1], pe[:, :, -1:] + 5], axis=2)
    changed = dsa_indexed._indexed_decode(q, qp, changed_latent, changed_pe, ids, None, 0.0625)
    assert_bits_equal(actual[:, :, :-1], changed[:, :, :-1])
    assert bool(mx.any(actual[:, :, -1] != changed[:, :, -1]).item())


@metal
@pytest.mark.parametrize("context", [23861, 24373, 28842])
@pytest.mark.parametrize("masked", [False, True])
def test_actual_hermes_decode_keeps_native_and_saved_bits(context, masked):
    from exlr8.model import Attention
    from exlr8.weights import SafeTensorFile
    path = Path(__file__).resolve().parents[1] / "artifacts/dsa-decode-indexed-inputs.safetensors"
    if not path.exists():
        pytest.skip("actual Hermes decode fixture is not present")
    file = SafeTensorFile(path)
    q, qp, latent, pe, ids, saved = [file.load(f"c{context}." + name) for name in
        ("q", "q_pe", "latent", "pe", "ids", "native_output")]
    mask = (mx.arange(context) % 7 != 6)[None, None, None] if masked else None
    args = (q.reshape(1, 64, 1, 512), qp.reshape(1, 64, 1, 64),
            latent.reshape(1, 1, context, 512), pe.reshape(1, 1, context, 64), ids.reshape(1, 1, 1, 2048), mask)
    attention = SimpleNamespace(scale=float(file.metadata["scale"]))
    attention._sparse_attention = lambda *values: Attention._sparse_attention(attention, *values)
    expected = attention._sparse_attention(*args)
    if not masked:
        assert_bits_equal(expected, saved.reshape(1, 64, 1, 512))
    dsa_indexed.install_indexed_prefill(attention)
    assert_bits_equal(attention._sparse_attention(*args), expected)
