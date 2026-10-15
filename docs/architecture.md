# Architecture

## Recurrent linear attention (authoritative equations)

Per layer, position `t`, dimension `d`, state `s in R^d` (fp32, `s_0 = 0`):

```
r_t = x_t @ W_r ;  k_t = x_t @ W_k ;  v_t = x_t @ W_v     (d x d projections)
alpha = exp(-softplus(w)) in (0,1)^d                      (learned decay)
s_t = alpha * s_{t-1} + k_t * v_t                         (elementwise)
y_t = sigmoid(r_t) * (s_t + u * k_t * v_t)                (u: learned bonus)
o_t = y_t @ W_o
```

No `T x T` matrix is ever built; state memory is `L*d` floats independent of
sequence length. Prefill, chunked processing, single-token decode, state
serialize/restore are supported; chunked and restored runs match full runs
when expert-capacity dropping does not intervene (see MoE).

## RMSNorm

`y = x / sqrt(mean(x^2) + eps) * g`, `eps=1e-6`, applied before attention and
before channel mixing in every block, plus a final norm before the tied head.
Same placement in training, prefill, and incremental inference.

## Channel mixing

- Dense blocks: gated FFN `silu(x@W_a) * (x@W_b) @ W_c` (3*d*hff params).
- MoE blocks: one **shared expert pool + shared router** across all blocks
  (documented choice bounding memory to a single pool). Expert:
  `gelu(x@W_in) @ W_out` (2*d*h params, exact per preset).

## Byte vocabulary

Token id = byte value for 0..255; BOS=256, EOS=257, PAD=258; vocab size 259.
Loss is causal next-byte cross-entropy reported in nats per predicted byte.

## Parameter arithmetic

Tied embedding/head. Per dense layer: `4*d^2 + 4*d + 3*d*hff`.
Total dense = `259*d + d + L*(...)`. MoE shared per layer: `4*d^2 + 4*d`
(attention + decay + bonus + 2 norms), plus embed, final norm, one `d x E`
router. See `docs/presets.md` for exact solved dimensions.
