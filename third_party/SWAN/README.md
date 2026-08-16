# SWAN: Semantic Watermarking with Abstract Meaning Representation

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

This repository contains the code for the ACL 2026 paper:

> **SWAN: Semantic Watermarking with Abstract Meaning Representation**

SWAN embeds watermark signatures into the semantic structure of sentences using Abstract Meaning Representation (AMR). Because the signature lives in the AMR graph, any paraphrase that preserves meaning automatically preserves the watermark.

## Repository Structure

```
.
├── amr_bank/                           # AMR Bank Creation (§3.1)
│   ├── analyze_amr_distribution.py     #   Build frequency distribution from MASSIVE-AMR corpus
│   ├── build_curated_amr_bank.py       #   Filter templates by frequency + node count → bank
│   ├── create_examples.py              #   Generate few-shot examples for the generation prompt
│   ├── display_top_templates.py        #   Display top-frequency templates
│   ├── banks/                          #   Pre-built curated AMR banks (50, 100, 500, 800 templates)
│   ├── artifacts/                      #   Intermediate outputs (full bank, examples, distributions,
│   │                                    #   pre-parsed human AMRs)
│   └── data/                           #   Place massive_amr.jsonl here (see Setup §4)
│
├── injection/                          # Watermark Injection (§3.2)
│   ├── watermark_generation.py         #   Main watermark injection script (Algorithm 1)
│   └── injection_amr_utils.py          #   Prompt construction, rejection sampling, S2match
│
├── detection/                          # Watermark Detection (§3.3)
│   ├── detect_from_parsed_amrs.py      #   Detection on pre-parsed AMRs (Algorithm 2)
│   ├── detect_end_to_end.py            #   End-to-end detection (parse + detect in one pass)
│   ├── detection_utils.py              #   Z-score computation, AUROC evaluation, ROC plotting
│   ├── parse_machine_text.py           #   GPU-parallel AMR parsing of machine-generated text
│   ├── parse_human_text.py             #   GPU-parallel AMR parsing of human text
│   └── evaluate_z_scores.py            #   Standalone evaluation from saved z-scores
│
├── evaluation/                         # Quality & Efficiency Evaluation (§4)
│   ├── text_quality_eval.py            #   LLM-as-judge text quality evaluation (§4.5)
│   ├── sampling_efficiency.py          #   Compute sampling efficiency stats (§4.3)
│   └── paraphrase_gen.py               #   Claude zero-shot paraphrase attack (§4.1)
│
├── utils/                              # Shared Utilities
│   ├── amr_utils.py                    #   AMR parse / normalize / template / S2match scoring
│   ├── s2match_patch.py                #   Our added S2match entry point — append to amr-metric-suite (see Setup §2)
│   ├── text_utils.py                   #   Prompt extraction and text processing
│   ├── bedrock_utils.py                #   AWS Bedrock API wrapper for LLM inference
│   └── load_c4_realnews_data.py        #   Download REALNEWS subset from C4
│
├── requirements.txt
├── LICENSE                             #   CC-BY-NC 4.0
└── README.md
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install S2match

SWAN uses S2match (Opitz et al., 2020) for soft semantic similarity between AMR
concepts. S2match lives in the [amr-metric-suite](https://github.com/flipz357/amr-metric-suite)
repo and is not pip-installable. We ship a small patch file
(`utils/s2match_patch.py`) that adds a single string-based entry point
(`compute_s2match_from_strings`) on top of the upstream code. Append it to the
upstream `s2match.py`:

```bash
git clone https://github.com/flipz357/amr-metric-suite.git
cat utils/s2match_patch.py >> amr-metric-suite/py3-Smatch-and-S2match/smatch/s2match.py
```

After this step the directory layout should be:

```
SWAN/
├── amr-metric-suite/
│   └── py3-Smatch-and-S2match/
│       └── smatch/
│           ├── amr_py3.py        # from upstream
│           ├── helpers.py        # from upstream
│           └── s2match.py        # upstream + our appended compute_s2match_from_strings
├── utils/
│   ├── s2match_patch.py          # the patch you appended — keep as reference
│   └── ...
└── ...
```

`utils/amr_utils.py` automatically adds `amr-metric-suite/py3-Smatch-and-S2match/smatch/`
to `sys.path` and imports `s2match` from there.

S2match also needs GloVe word vectors:

```bash
mkdir -p vectors
cd vectors
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ..
export GLOVE_VECTORS_PATH="vectors/glove.6B.100d.txt"
```

### 3. Download the amrlib parsing model

SWAN uses [amrlib](https://github.com/bjascob/amrlib) for AMR parsing.
The parsing and generation models need to be downloaded manually following the
[amrlib model installation guide](https://amrlib.readthedocs.io/en/latest/install/#install-the-models):

- `model_parse_xfm_bart_large-v0_1_0` — BART-large AMR parser (used for detection)
- `model_generate_t5wtense-v0_1_0` — T5 AMR-to-text generator

Download both models from the links on the [amrlib models page](https://amrlib.readthedocs.io/en/latest/install/#install-the-models),
then extract them into amrlib's data directory and create the required symlinks:

```bash
# Find amrlib's data directory and create it if needed
AMRLIB_DATA=$(python -c "import amrlib; import os; print(os.path.join(os.path.dirname(amrlib.__file__), 'data'))")
mkdir -p $AMRLIB_DATA

# Move downloaded model archives to amrlib's data directory
mv model_parse_xfm_bart_large-v0_1_0.tar.gz $AMRLIB_DATA/
mv model_generate_t5wtense-v0_1_0.tar.gz $AMRLIB_DATA/

# Extract and symlink the parser model
cd $AMRLIB_DATA
tar xzf model_parse_xfm_bart_large-v0_1_0.tar.gz
ln -snf model_parse_xfm_bart_large-v0_1_0 model_stog

# Extract and symlink the generation model
tar xzf model_generate_t5wtense-v0_1_0.tar.gz
ln -snf model_generate_t5wtense-v0_1_0 model_gtos
```

Verify the parser loads correctly:

```bash
cd -  # Return to the project root directory
python -c "import amrlib; amrlib.load_stog_model(); print('Parser loaded successfully')"
```

### 4. Download the MASSIVE-AMR corpus (optional — for rebuilding the AMR bank)

The pre-built AMR banks are included in `amr_bank/banks/`. If you want to rebuild
them from scratch, download the [MASSIVE-AMR](https://github.com/amazon-science/MASSIVE-AMR)
corpus and place it in `amr_bank/data/`:

```bash
mkdir -p amr_bank/data
wget -O amr_bank/data/massive_amr.jsonl \
    https://raw.githubusercontent.com/amazon-science/MASSIVE-AMR/main/data/massive_amr.jsonl
```

### 5. Download the REALNEWS evaluation dataset

```bash
python utils/load_c4_realnews_data.py --k 250
```

This downloads the first 250 examples from the REALNEWS subset of C4 and saves
them to `data/c4-val` (prompts for generation) and `data/c4-human` (human reference for detection).

### 6. AWS Bedrock access (for injection and evaluation)

Watermark injection and text quality evaluation use AWS Bedrock for LLM inference.
Configure your AWS credentials:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

> **Note on reproducibility.** The defaults in this repo point to Claude Sonnet 4.5
> (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`). The paper results were produced
> with Claude 3.5 Sonnet v2 (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`) for
> injection and Claude 3.7 Sonnet (`anthropic.claude-3-7-sonnet-20250219-v1:0`) for
> paraphrase attacks and text quality evaluation. Those model versions have since
> been retired from Bedrock. Pass `--model_id` explicitly if you want to pin to a
> specific available model.

Alternatively, you can use HuggingFace models locally (e.g., DeepSeek-R1-Distill-Qwen-14B)
by passing `--hf_model True --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.

## Reproducing the Paper Results

### Step 1: Build the AMR Bank (§3.1)

Pre-built banks are included in `amr_bank/banks/` (50, 100, 500, 800 templates).
To rebuild from scratch:

```bash
# Analyze template distribution from MASSIVE-AMR
python amr_bank/analyze_amr_distribution.py \
    --input_jsonl amr_bank/data/massive_amr.jsonl \
    --output_dir amr_bank/artifacts

# Build curated banks of different sizes
python amr_bank/build_curated_amr_bank.py \
    --distribution_json amr_bank/artifacts/normalized_template_amr_distribution.json \
    --bank_size 50 --output_dir amr_bank/banks \
    --output_name amr_bank_50.json

# Generate few-shot examples for the injection prompt
python amr_bank/create_examples.py \
    --input_jsonl amr_bank/data/massive_amr.jsonl \
    --output_json amr_bank/artifacts/normalized_amr_examples.json
```

### Step 2: Watermark Injection (§3.2)

Generate watermarked text using the instruction-based sampling approach:

```bash
python injection/watermark_generation.py \
    --data data/c4-val \
    --amr_bank_path amr_bank/banks/amr_bank_50.json \
    --amr_examples_path amr_bank/artifacts/normalized_amr_examples.json \
    --model_id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --region us-east-1 \
    --num_sentences 5 \
    --max_trials 50 \
    --output_dir output/watermarked
```

To generate without watermarking (for baseline comparison):

```bash
python injection/watermark_generation.py \
    --data data/c4-val \
    --amr_bank_path amr_bank/banks/amr_bank_50.json \
    --amr_examples_path amr_bank/artifacts/normalized_amr_examples.json \
    --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --hf_model True \
    --no_watermark \
    --output_dir output/no_watermark
```

### Step 3: Watermark Detection (§3.3)

**Two-stage detection** (recommended for large-scale evaluation — separates GPU parsing from CPU detection):

```bash
# Stage 1: Parse machine-generated text to AMRs (GPU)
python detection/parse_machine_text.py output/watermarked \
    --gpu_ids 0,1,2,3 \
    --batch_size 32

# Stage 1: Parse human text to AMRs (GPU)
python detection/parse_human_text.py \
    --dataset_path data/c4-human \
    --gpu_ids 0,1,2,3 \
    --batch_size 32

# Stage 2: Run detection on pre-parsed AMRs (CPU)
python detection/detect_from_parsed_amrs.py \
    --parsed_machine output/watermarked/parsed_amrs.json \
    --parsed_human amr_bank/artifacts/human_parsed_amrs_250.json \
    --amr_bank_path amr_bank/banks/amr_bank_50.json \
    --threshold 0.65 \
    --lmbd 0.25 \
    --output_dir results/detection
```

**End-to-end detection** (parsing + detection in one pass):

```bash
python detection/detect_end_to_end.py output/watermarked \
    --human_text data/c4-human \
    --amr_bank_path amr_bank/banks/amr_bank_50.json \
    --threshold 0.65 \
    --lmbd 0.25
```

### Step 4: Paraphrase Attacks (§4.1)

Generate paraphrased versions of watermarked text using Claude Sonnet 4.5:

```bash
python evaluation/paraphrase_gen.py \
    --dataset_path output/watermarked \
    --output_dir output/watermarked_paraphrased \
    --model_id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --region us-east-1
```

Then re-run detection on the paraphrased text:

```bash
python detection/parse_machine_text.py output/watermarked_paraphrased \
    --paraphrased \
    --gpu_ids 0,1,2,3

python detection/detect_from_parsed_amrs.py \
    --parsed_machine output/watermarked_paraphrased/parsed_amrs.json \
    --parsed_human amr_bank/artifacts/human_parsed_amrs_250.json \
    --amr_bank_path amr_bank/banks/amr_bank_50.json \
    --output_dir results/detection_paraphrased
```

### Step 5: Text Quality Evaluation (§4.5)

```bash
python evaluation/text_quality_eval.py \
    --dataset_path output/watermarked \
    --output_path results/quality_scores.json \
    --model_id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --region us-east-1
```

### Step 6: Sampling Efficiency (§4.3)

```bash
python evaluation/sampling_efficiency.py \
    --sampled_data_paths output/watermarked/sampled_data.json
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `s2match_threshold` (injection) | 0.75 | S2match score required to accept a generated sentence |
| `threshold` (detection) | 0.65 | S2match score required to flag a sentence as "green" |
| `lmbd` | 0.25 | Expected fraction of green sentences under the null hypothesis |
| `max_trials` | 50 | Maximum rejection sampling attempts per sentence |
| `max_templates_to_try` | 10 | Maximum different AMR templates to try per sentence |
| `max_retries_per_template` | 5 | Maximum retries per template before switching |
| `bank_size` | 50 | Number of AMR templates in the bank |

## License

This code is released under the [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@inproceedings{ye2026swan,
  title     = {SWAN: Semantic Watermarking with Abstract Meaning Representation},
  author    = {Ye, Ziping and Dey, Gourab and Christodoulopoulos, Christos and
               Peris, Charith and Ramakrishna, Anil and Ruan, Weitong and
               Galstyan, Aram and Chang, Kai-Wei and Gupta, Rahul and Mehrabi, Ninareh},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026},
  address   = {San Diego, California, USA},
  publisher = {Association for Computational Linguistics}
}
```
