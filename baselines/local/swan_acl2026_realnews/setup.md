# SWAN Baseline Local Setup

## Verified environment

- Date: 2026-07-22 (Asia/Shanghai)
- OS: Windows 10 build 26200
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- Driver: 580.88; driver CUDA API 13.0
- Local CUDA toolkit: 12.8
- Python: CPython 3.11.14 in `.venv`
- PyTorch: 2.4.1+cu124
- SWAN commit: `2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a`
- amr-metric-suite commit: `711a231d3600139662fe048460e10d773ff8e214`

## Downloaded resources

- GloVe 6B archive SHA-256: `617AFB2FE6CBD085C235BAF7A465B96F4112BD7F7CCB2B2CBD649FED9CBCF2FB`
- BART-large parser archive SHA-256: `2FCFE2113527ACCB752B6C574994022B3375B66D59DA5042356C0F853E879849`
- T5-with-tense generator archive SHA-256: `E13A57FF86C03483686010BA8FDEBB61EE211561EA1E5793DD557F8FACAC4EEF`
- C4 RealNews: 250 `validation` prompts and 5000 `train` human references, obtained through the official SWAN streaming script.

## Compatibility patches

1. Applied SWAN's `compute_s2match_from_strings` entry point to S2MATCH.
2. Read GloVe explicitly as UTF-8 on Windows.
3. Cache GloVe once per detector process.
4. Load the BART parser lazily so CPU detector workers do not allocate it.
5. Route `deepseek-*` IDs to DeepSeek ChatCompletions in non-thinking mode.
6. Added `--max_samples` and `--seed`; fixed API-mode no-watermark generation.

These patches do not change the detector threshold, null rate, AMR bank, parser weights, or S2MATCH scoring formula.

## Smoke evidence

- CUDA tensor operation: passed.
- BART parser: passed; one sentence parsed; peak allocated VRAM 1615 MiB.
- T5 generator: passed; one AMR generated; peak allocated VRAM 862 MiB.
- S2MATCH: identical AMR 1.0; comparison AMR 0.7974604787.
- Local detector: 10 human documents, 47 sentences, 50-template bank, threshold 0.65, lambda 0.25; elapsed scoring time 12.680 s.
- Detector output: `runs/smoke_10/human_detection.json`.

## Required runtime variables

```powershell
$env:HF_HOME = "$PWD\.hf-cache"
$env:NLTK_DATA = "$PWD\.nltk_data"
$env:GLOVE_VECTORS_PATH = "$PWD\third_party\SWAN\vectors\glove.6B.100d.txt"
$env:DEEPSEEK_API_KEY = "<set locally; never commit>"
```

## Next gate

The local detector smoke is accepted. Before the 30–50 prompt pilot, run one paid DeepSeek API request and record the returned model identifier and call date. No API key was present or tested when this file was written.
