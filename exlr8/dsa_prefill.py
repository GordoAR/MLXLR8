"""DSA attention candidates. The model keeps its tested attention path."""

import mlx.core as mx


def full_latent_attention(q, q_pe, latent, k_pe, indices, mask, scale, chunk=64):
    """Reuse the full latent K/V matrix with an exact selected-key mask."""
    batch, _, length, _ = q.shape
    key_length = latent.shape[2]
    outputs = []
    keys = mx.arange(key_length)
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        allowed = mx.ones((batch, 1, end - start, key_length), dtype=mx.bool_)
        if indices is not None:
            allowed = mx.put_along_axis(
                mx.zeros_like(allowed), indices[:, :, start:end, :],
                mx.array(True), axis=-1,
            )
        positions = mx.arange(key_length - length + start, key_length - length + end)
        allowed &= keys[None, None, None, :] <= positions[None, None, :, None]
        if mask is not None:
            allowed &= mx.broadcast_to(mask, (batch, 1, length, key_length))[:, :, start:end, :]
        bias = (q_pe[:, :, start:end, :] * scale) @ k_pe.swapaxes(-1, -2)
        bias = mx.where(allowed, bias, -float("inf"))
        outputs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, start:end, :], latent, latent, scale=scale, mask=bias,
        ))
    return mx.concatenate(outputs, axis=2)


def _gathered_rows(q, q_pe, latent, k_pe, indices, mask, scale, chunk):
    batch, heads, length, _ = q.shape
    key_length, selected = latent.shape[2], indices.shape[-1]
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        rows = end - start
        idx = indices[:, 0, start:end, :]

        def gather(source):
            dims = source.shape[-1]
            return mx.take_along_axis(
                mx.broadcast_to(source, (batch, rows, key_length, dims)),
                mx.broadcast_to(idx[..., None], (batch, rows, selected, dims)), axis=2,
            ).reshape(batch * rows, 1, selected, dims)

        lr, pr = gather(latent), gather(k_pe)
        qr = q[:, :, start:end, :].swapaxes(1, 2).reshape(batch * rows, heads, 1, -1)
        qpr = q_pe[:, :, start:end, :].swapaxes(1, 2).reshape(batch * rows, heads, 1, -1)
        bias = (qpr * scale) @ pr.swapaxes(-1, -2)
        positions = mx.arange(key_length - length + start, key_length - length + end)
        valid = idx <= positions[None, :, None]
        if mask is not None:
            row_mask = mx.broadcast_to(mask, (batch, 1, length, key_length))[:, 0, start:end, :]
            valid &= mx.take_along_axis(row_mask, idx, axis=-1)
        bias = mx.where(valid.reshape(batch * rows, 1, 1, selected), bias, -float("inf"))
        yield qr, lr, bias, rows


def split_heads_attention(q, q_pe, latent, k_pe, indices, mask, scale,
                          chunk=32, head_chunk=32, force_fused=False):
    """Divide query heads for vector SDPA; keep the current sparse gathers.

    D512 fused calls require MLX_SDPA_D512_MIN_KL=0 in the caller's process.
    This helper does not change that environment variable.
    """
    batch, heads = q.shape[:2]
    outputs = []
    for qr, lr, bias, rows in _gathered_rows(q, q_pe, latent, k_pe, indices, mask, scale, chunk):
        parts = []
        for head in range(0, heads, head_chunk):
            stop = min(head + head_chunk, heads)
            parts.append(mx.fast.scaled_dot_product_attention(
                qr[:, head:stop, :, :], lr, lr, scale=scale,
                mask=bias[:, head:stop, :, :], force_fused=force_fused,
            ))
        output = mx.concatenate(parts, axis=1)
        outputs.append(output.reshape(batch, rows, heads, -1).swapaxes(1, 2))
    return mx.concatenate(outputs, axis=2)


def heads_as_rows_attention(q, q_pe, latent, k_pe, indices, mask, scale, chunk=32):
    """Use the independent query heads as matrix rows to reuse sparse K/V."""
    batch, heads = q.shape[:2]
    outputs = []
    for qr, lr, bias, rows in _gathered_rows(q, q_pe, latent, k_pe, indices, mask, scale, chunk):
        output = mx.fast.scaled_dot_product_attention(
            qr.swapaxes(1, 2), lr, lr, scale=scale, mask=bias.swapaxes(1, 2),
        )
        outputs.append(output.reshape(batch, rows, heads, -1).swapaxes(1, 2))
    return mx.concatenate(outputs, axis=2)
