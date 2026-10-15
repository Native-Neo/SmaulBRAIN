# Presets (exact, machine-checked)

| Preset     | d    | Layers | FFN hidden | Experts | Top-k | Expert hidden |
|------------|------|--------|------------|---------|-------|---------------|
| moe-full   | 512  | 6      | —          | 64      | 8     | 5000          |
| moe-tiny   | 64   | 4      | —          | 32      | 2     | 80            |
| dense-full | 1000 | 19     | 3152       | —       | —     | —             |
| dense-tiny | 32   | 15     | 128        | —       | —     | —             |

Exact counts asserted at model construction (`Model._assert_counts`) and in
`tests/test_presets.py`:

- `moe-full`: 2*512*5000 = **5,120,000** per expert; shared 6,469,632; whole 334,149,632.
- `moe-tiny`: 2*64*80 = **10,240** per expert; shared 85,248; whole 412,928.
- `dense-full`: **256,000,000** total (tied head).
- `dense-tiny`: **256,000** total (tied head).

Growth preserves per-expert architecture and top-k; it never redefines presets.
After N events (no pruning): full 64+8N experts, tiny 32+2N.
