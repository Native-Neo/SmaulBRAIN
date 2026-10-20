# Default NCM runtime

The `train.py` and `infer.py` entry points use this Next-Concept Modeling
(NCM) adaptation by default. It is inspired by the paper and official
[HazyResearch/ncm repository](https://github.com/HazyResearch/ncm), but remains
a compact implementation rather than a reproduction of the paper's full recipe.
NCM checkpoints use `tokenizer.pt` and `generator.pt` rather than the legacy
SmaulBRAIN `manifest.json` format.

## What the architecture does

1. **Train a multiscale residual-VQ tokenizer.** Each level has its own codebook
   and a longer code sequence than the preceding level. Quantized residuals are
   upsampled and accumulated; the decoder learns to reconstruct the byte stream.
2. **Freeze the tokenizer.** The generator uses its codebooks to reconstruct the
   cumulative representation of already-known levels.
3. **Train a SmaulBRAIN-style recurrent generator.** Each level's input is the
   resized cumulative reconstruction of *earlier* levels, never its own target
   IDs. The generator uses recurrent linear attention with learned decay and
   bonus, RMSNorm, residual connections, and a shared sparse router/expert pool. A
   chunked vectorized scan computes recurrent states without a time-by-time
   attention matrix. Each level has an independent recurrent state; prompt
   features are repeated for each level to prevent information leaking between
   levels. Conditioning corruption changes input conditioning IDs, not labels.
The tokenizer now applies RMSNorm to the learned byte features and the
cumulative reconstruction before its byte decoder. Training logs reconstruction
cross-entropy separately from the averaged vector-quantization/commitment loss,
so a rising total can be traced to its component instead of appearing as one
opaque number. This is a stabilization measure to evaluate, not a guarantee
that the tokenizer has converged.

4. **Generate coarse to fine.** Each level costs one sequential generator call;
   all positions in that level are predicted in parallel. The frozen tokenizer
   decoder maps the final hierarchy back to byte IDs. Prefix-conditioned
   continuation is supported.

The generator is a **bidirectional adaptation** of SmaulBRAIN's recurrent
attention: it scans each known level both forward and backward, so every
position can use same-level context while all codes in that level are predicted
in parallel. The standard autoregressive model uses a causal forward state.
For MoE presets, one router and expert pool are shared across every recurrent
block. Dense presets use a per-layer SwiGLU channel mixer. Preset dimensions
are inherited from the original four preset definitions, but NCM tokenizer and
hierarchy parameters change the total parameter counts.

The NCM linear layers support FP32, software FP8 E4M3 emulation, and packed
ternary operations. With `--forward-quant ternary`, linear matrix weights are
persisted as packed 2-bit ternary codes (four values per byte). They are not
held in FP32 master-weight tensors, and the trainer does not accumulate dense
FP32 weight-gradient tensors for those matrices. Forward and input-gradient
matrix products use the packed ternary kernels. Backpropagation calculates a
transient dense gradient signal, then queues sparse gradient-sign updates that
modify the packed codes directly after backward. Biases, embeddings, codebooks,
and normalization parameters remain floating point.

This direct-discrete ternary update rule is experimental and is not conventional
straight-through-estimator QAT. Its convergence and generated-text quality must
be evaluated on real runs. Legacy FP32-master ternary checkpoints are converted
to packed ternary weights on load; optimizer moments for the converted matrices
are reset.

The prototype is intentionally smaller than the paper's VQ-VAE and generator.
It does not yet reproduce the paper's learned quantizer schedules, exact
residual projections, full corruption schedule, training recipe, or evaluation
protocol. Its results must not be presented as reproducing the paper's reported
speedups.

## Train

Install the repository's dependencies and, for Parquet data, pyarrow:

```bash
pip install -r requirements.txt
pip install pyarrow
```

The supported presets are `dense-tiny`, `dense-full`, `moe-tiny`, and
`moe-full`. `train.py` is the default entry point; it trains the tokenizer
and generator and uses SmaulOPT:

```bash
python train.py --preset moe-tiny --corpus ./data/corpus.bin \
  --steps 100 --checkpoint checkpoints/ncm-run
python train.py --preset moe-tiny --corpus ./data/corpus.bin \
  --steps 100 --checkpoint checkpoints/ncm-ternary \
  --forward-quant ternary --backward-mode ternary
python infer.py --checkpoint checkpoints/ncm-run \
  --prompt "Once upon a time, " --max-new-tokens 128
```

Direct experiments remain available through `ncm_train.py`, where `--preset
custom` allows small custom dimensions for fast tests and profiling.

The reader streams bounded chunks and does not load the entire corpus into RAM.
For Parquet files it prefers columns named text, content, response, output,
completion, answer, prompt, instruction, messages, or conversation. If none
exist, it reads the available columns as text. This generic row-to-text
conversion is a convenience, not a replacement for a carefully curated
training-data pipeline.

The tokenizer is trained first. The generator then learns continuation by
conditioning on the first half of each sampled byte window and predicting the
second half. Stage checkpoints are saved under the output directory, and
training resumes automatically from the latest stage checkpoint. Run the two
stages separately with --stage tokenizer and --stage generator if needed; the
generator stage requires a matching tokenizer.pt.

Defaults are deliberately small for low-memory experimentation. Increase
--d-model, --layers, --levels, --codebook-size, or --experts only after a small
run fits your RAM/VRAM. --top-k must not exceed --experts. The old --heads
argument is accepted for command compatibility but has no effect on recurrent
blocks.

## Benchmark

```bash
python ncm_bench.py --checkpoint-dir ./ncm-recurrent-checkpoints \
  --prompt "Once upon a time, " --new-bytes 128 --runs 5
```

NCM checkpoints are separate from the legacy SmaulBRAIN checkpoint
format and from older Transformer-based NCM generator checkpoints. Use a fresh
checkpoint directory when changing model preset or architecture. Quantization
mode can be selected at training or inference; changing it does not change
parameter shapes, but ternary/FP8 arithmetic may change outputs.

This reports end-to-end latency including multilevel generation and decoding.
For a meaningful comparison against the existing autoregressive SmaulBRAIN
model, use the same hardware, prompt, generated byte count, and batch size; also
compare held-out quality. The reported training loss is codebook-ID cross-entropy,
not the existing model's next-byte cross-entropy, so the numeric loss values
cannot be directly compared. Include tokenizer training time, generator training
time, peak memory, and codes per byte. A faster NCM sample is not a fair win if
it produces worse text or requires much more training.

## Tests

```bash
python -m pytest -q tests/test_ncm.py
```
