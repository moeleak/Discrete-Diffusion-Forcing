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
- A KV-retrieval scoring ablation must use the fixed first 100 ordered
  long-page sample IDs and keep the image tiles, overview, operation-only
  query, Top-K, checkpoint, YaRN/position policy, prompt, seed, decoding, OCR,
  and disabled token/head KV compression identical. Run dense/no retrieval,
  legacy causal next-token scoring, and bidirectional masked scoring serially
  on one GPU. The legacy causal scorer is ablation-only; the default runtime
  remains bidirectional masked scoring. Audit target-tile recall only after
  inference from ground-truth boxes.
- A KV-retrieval attention-direction ablation must use the fixed first 100
  ordered long-page sample IDs and keep the image tiles, overview,
  operation-only query token IDs and positions, complementary corruption
  masks, same-position targets, mask rounds, Top-K, checkpoint,
  YaRN/position policy, prompt, seed, decoding, OCR, sequential scoring, and
  disabled token/head KV compression identical. Only the query attention
  topology may differ: causal masked query versus fully bidirectional masked
  image-query scoring. Do not use the legacy clear-query next-token scorer as
  either arm of this controlled comparison.
- A KV-retrieval Top-K ablation must use the fixed first 100 ordered long-page
  sample IDs and keep the bidirectional masked scorer, sequential scoring,
  operation-only query, complementary masks, overview, checkpoint,
  YaRN/position policy, image tile size, prompt, seed, decoding, OCR, resident
  KV capacity, and disabled token/head KV compression identical. Only
  `KV_RETRIEVAL_TOPK_IMAGES` may change; the forced overview is additional to
  that source-tile count.
