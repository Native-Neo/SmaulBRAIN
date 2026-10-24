# Open questions

Items below are intentionally not implemented rather than guessed at. See
`PORT_NOTES.md` for parity results and checkpoint deviations.

1. `--pagingmthd`, adaptive halting depth, and `--trunk-lr-mult` do not exist in
   this repository's source or CLI. No behavior was invented for them.
2. The default NCM architecture does not implement MTP (`--mtp-depth > 0` is
   rejected), adaptive halting, expert growth, or expert pruning. It uses a
   fixed in-memory expert pool.
3. NCM paging is `R` only. CUDA/VRAM paging, CUDA execution, and hardware FP8 are
   unavailable in the CPU-only build.
4. `--stop-eos` is rejected because the NCM byte tokenizer has no EOS token.
   `--prefill-chunk > 0` is rejected because the encoder processes the full
   prefix at once.
5. Legacy PyTorch NCM `.pt` checkpoints require the development-only converter
   in `tests/_reference/`. Loading legacy AdamW optimizer state is not possible;
   model weights and the step are preserved and fresh SmaulOPT state is used.
6. PyTorch RNG streams and PyTorch-sampled ternary update trajectories are not
   reproduced. Greedy generation is deterministic, but sampled runs differ.
7. Full-preset convergence, memory use, and throughput on the target Ivy Bridge
   CPU have not been measured.
