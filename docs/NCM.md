# Experimental NCM path

SmaulBRAIN's normal byte-level recurrent model is unchanged. This folder-level
experiment adds the two core pieces needed to test Next-Concept Modeling (NCM):

1. `ResidualVQTokenizer` learns multiscale residual code sequences from bytes.
   The first code level is shortest; later levels become progressively finer.
   Its reconstruction objective trains the tokenizer before it is frozen.
2. `NCMTransformer` predicts code IDs at every level in one training forward
   call. Its attention mask prevents a level's target codes from leaking into
   that level's predictions. At inference, `generate_codes` fills one level
   per sequential step and predicts every position in that level in parallel.

This is an experimental baseline inspired by NCM, not a claim of exact parity
with the paper's tokenizer, training recipe, or reported performance. It is not
wired into the default `train.py` / `infer.py` path and does not change existing
checkpoints.

## Quick smoke test

Install the existing requirements, then run:

```bash
python -m pytest -q tests/test_ncm.py
```

## Minimal training sketch

Tokenizer stage (train separately; freeze after convergence):

```python
tokenizer = ResidualVQTokenizer(config)
logits, loss, code_ids = tokenizer(byte_batch)
loss.backward()
optimizer.step()
```

Generator stage:

```python
for parameter in tokenizer.parameters():
    parameter.requires_grad_(False)
code_ids = tokenizer.encode(byte_batch)
loss = ncm_loss(generator, code_ids)
loss.backward()
optimizer.step()
```

For a valid speed comparison, benchmark this path and the existing model on the
same corpus, hardware, output length, and quality target. Measure tokenizer
compression (codes per byte), generator training time, sequential generation
steps, wall-clock output speed, and peak memory. The tokenizer's encode/decode
cost must be included; fewer hierarchy steps alone do not guarantee a win.
