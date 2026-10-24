# Torch → NumPy port plan (SmaulBRAIN)

Inventory from grepping `torch`, `nn.`, `F.`, `autograd`, `.cuda`, `.to(`,
`DataLoader`, `torch.save/load`. Runtime PyTorch is removed from this repo; the
PyTorch oracle lives only in `tests/_reference/`.

## Shared layer

`SmaulBRAIN` already depends on the `smaulbrain` package (quantization / presets).
The NumPy autograd layer (`smaulbrain.npgrad`) and the safetensors I/O live in
`smaulbrain-core`; this repo consumes them. No second autograd implementation is
introduced.

## File / symbol inventory

### `ncm.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `NCMConfig` | dataclass only | unchanged |
| `_NCMQuantizedLinear` | `torch.autograd.Function` | `npgrad.make_op` custom forward/backward |
| `_DirectTernaryLinear` | autograd Function + packed ternary | `npgrad.make_op`, pending update queue |
| `NCMLinear` | `nn.Linear`, buffers, `_version` | `npgrad.Module` + `Parameter` + uint8 packed buffers |
| `migrate_ternary_state_dict` | torch state dict | NumPy state dict |
| `apply_direct_ternary_updates` | module traversal | module traversal, NumPy segments |
| `resize_sequence` | `F.interpolate` | `npgrad.interpolate1d` |
| `RMSNorm` | `nn.Module`, `rsqrt` | `npgrad.Module`, `npgrad.rsqrt` |
| `SmaulExpert` | `nn.Module`, `F.gelu` | `npgrad.Module`, `npgrad.gelu` |
| `SmaulSharedMoE` | `topk/softmax/argsort/bincount/index_add_` | NumPy grouping + `npgrad.index_add_rows` |
| `_NCMNativeRecurrentScan` | autograd Function over C++ scans | `npgrad.scan` (chunked closed form + backward) |
| `SmaulRecurrentBlock` | `flip/cat/sigmoid/softplus` | `npgrad` equivalents |
| `SmaulDenseFFN` | `F.silu` | `npgrad.silu` |
| `ResidualVQTokenizer` | `nn.Embedding`, `adaptive_avg_pool1d`, `mse_loss`, `cross_entropy` | `npgrad.Embedding`, `npgrad.adaptive_avg_pool1d`, `npgrad.mse_loss`, `npgrad.cross_entropy` |
| `_quantize` | `argmin`, codebook seeding, straight-through | NumPy argmin; in-place codebook copy; `detach()` |
| `NCMGenerator` | module composition, checkpointing, `torch.split/cat` | `npgrad` `split/cat`; checkpointing flag retained (recompute path uses no-graph rerun) |
| `generate_codes/generate` | `argmax/multinomial/topk/sort` | NumPy sampling with `npgrad.default_rng` |
| `ncm_loss` | `F.cross_entropy`, tensor weights | `npgrad.cross_entropy` |
| `backward_ncm_loss_chunked` | `.backward()` per level | `Tensor.backward()` per level |

### `ncm_train.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `ByteBatchStream.next` | `torch.frombuffer/stack` | `np.frombuffer` + `np.stack` (int64) |
| `SmaulTrainingOptimizer` | `named_parameters`, `.grad`, `.zero_` | `Module` API, `Parameter.grad` |
| `save_checkpoint` | `torch.save` | safetensors (config JSON in header metadata) |
| `load_stage_checkpoint` | `torch.load`, `load_state_dict` | safetensors reader + `load_state_dict` |
| `architecture_parameter_count` | `.numel()` | `np.size` / logical packed size |
| `main` | `torch.manual_seed`, `.to(device)`, clip_grad_norm_ | `npgrad.manual_seed`, no-op `.to`, `npgrad.clip_grad_norm_` |

### `train.py`
No direct torch import; passes `--device` through. Defaults to CPU; no CUDA
probing.

### `infer.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| checkpoint load | `torch.load` | safetensors reader |
| prompt | `torch.tensor` | `np.asarray` |
| generation | `torch.inference_mode` | `npgrad.no_grad` |
| output | `.tolist()` | `.tolist()` |

### `ncm_bench.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| checkpoint load | `torch.load` | safetensors reader |
| device / inference mode | `torch.device`, `torch.inference_mode` | CPU string, `npgrad.no_grad` |
| CUDA timing / memory | `torch.cuda.*` | removed (CPU-only target) |

### `bench.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| param init | `torch.Generator`, `build_params` | `npgrad.default_rng`, NumPy params |
| token sampling | `torch.randint` | `numpy.random.Generator.integers` |
| quant cache bytes | `numel*element_size` | `nbytes` |

### `bench_router.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| routing reference | torch sort/softmax/keep | NumPy reference |
| timing | `torch.tensor(...).median` | `statistics.median` |

### `evaluate.py` / `prune.py`
No torch imports; adapt to NumPy checkpoints and `np.isfinite`.

### Tests
All existing torch tests move to `tests/_reference/`. New runtime tests cover
NumPy training, inference, checkpoint round-trip, paging, and parity (dev oracle
in `tests/_reference/`).

## Checkpoint compatibility

* New NCM stage files: `tokenizer.safetensors`, `generator.safetensors`,
  `tokenizer-training.safetensors`, `generator-training.safetensors`.
* Config is stored in the safetensors `__metadata__` JSON (and read back into
  `NCMConfig`).
* Legacy `.pt` files are converted with
  `tests/_reference/convert_torch_checkpoint.py`.

## Numerical notes

* fp32 runtime; float64 accumulations where the original used `.double()`.
* PyTorch RNG streams differ; loaded weights are exact.
* `F.interpolate(..., mode="linear", align_corners=False)` and `area` are
  reimplemented and parity-tested.

## Open questions

1. `--pagingmthd`, adaptive halting depth, and `--trunk-lr-mult` do not exist in
   this repository; nothing is invented for them.
2. `--device cuda` is accepted for CLI compatibility but executes on CPU.
