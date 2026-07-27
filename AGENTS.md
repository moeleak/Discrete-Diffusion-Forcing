# Repository Workflow

- Work directly on `main`; do not create feature or topic branches.
- Commit every repository modification using the Conventional Commits format.
- Push completed commits directly to `main`.
- Before attributing a quality change to YaRN, run the 100-sample isolation
  benchmark. Both arms must use the same sample IDs, checkpoint-native resized
  image, native multimodal positions, prompt, seed, decoding parameters,
  16K-resident KV capacity, and disabled KV compression; only RoPE scaling and
  its advertised maximum position may differ.
- Do not launch more than 100 benchmark samples unless the user explicitly
  authorizes the larger run.
- A GUI tile-size ablation means the full-page screenshot tile edge
  (`FULL_PAGE_TILE_SIZE`), not the D2F diffusion block size. Keep the model,
  sample IDs, prompt instruction, overview/OCR policy, RoPE, D2F block size,
  decoding, and KV policy fixed; use 14px-aligned tile sizes and regenerate
  the prompt's runtime image count.
- A causal true-long YaRN comparison must use the same full-page tiles,
  sequential positions above 16K, 128K model limit, 65,536-token resident KV
  capacity, seed, prompt, and decoding parameters in both arms. Compare
  unscaled 128K against YaRN 128K; do not present native-resized original 16K
  versus full-page YaRN as a controlled YaRN experiment.
