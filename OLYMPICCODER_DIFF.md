# OlympicCoder-7B vs. our `configs/train/olympic_7b.py`

Every difference between open-r1's OlympicCoder-7B SFT and our reproduction, as of
2026-10-23. "Theirs" means **the run that produced the published weights**, not the
recipe file: the two disagree, and the artifacts shipped with the model settle it.

## Sources, and which one wins when they conflict

| source | what it says | trust |
|---|---|---|
| `open-r1/OlympicCoder-7B/training_args.bin` (pickled `SFTConfig` of the actual run) | lr **4e-5**, `max_grad_norm` **1.0**, `use_liger_kernel` **False**, batch 2 × accum 8, `packing` False, `max_length` 32768, `warmup_ratio` 0.03, cosine_with_min_lr 0.1, wd 0, `do_eval` False, `save_total_limit` 1, seed 42, `attn_implementation` flash_attention_2, `torch_dtype` bfloat16 | **authoritative** |
| `open-r1/OlympicCoder-7B/trainer_state.json` | 3,170 steps, 10 epochs, peak lr 4e-5 → 4e-6, loss 1.02 → 0.14, grad_norm 2.06 at step 1 | authoritative |
| blog `open-r1/update-3` | lr 4e-5 ("each doubling gained almost 10 points on LiveCodeBench"), 10 epochs, batch 128, C++ only, packing hurts, prefill `<think>` | consistent with the artifacts |
| model card README | lr 4e-5, batch 2 × 8 on 8 devices | consistent |
| `recipes/OlympicCoder-7B/sft/config_v00.00.yaml` (both revisions, 2025-03-13 and 2025-04-11) | lr **1e-5**, `max_grad_norm` **0.2**, `use_liger_kernel` **true** | **wrong on all three** — the recipe never matched the run |

Our config currently follows the recipe file for exactly the three values the
recipe gets wrong.

## Training

| | OlympicCoder-7B (actual) | ours (`olympic_7b.py`, job 24931873 + resume) | effect |
|---|---|---|---|
| base model | Qwen/Qwen2.5-Coder-7B-Instruct | same | — |
| learning rate (peak) | **4e-5** | **1e-5** | 4× smaller steps for the whole run |
| `max_grad_norm` | **1.0** | **0.2** | at the observed starting grad norm of ~2 (theirs 2.06, ours ~2), theirs clips to 0.5×, ours to 0.1×. **Smaller than it looks**: Adam divides by √v, so a uniform rescaling of the gradient is mostly absorbed — the step stays ≈ lr. The clip matters only through ε and through steps where it is active on some micro-batches and not others. (Round three correction; an earlier draft called this a 20× difference.) |
| epochs | 10 | 10 | — |
| effective batch | 128 (2 per device × 8 accum × 8 GPUs) | 128 (1 × 16 × 8) | same size; their micro-batches of 2 are padded to the longer sequence, ours are never padded |
| schedule | cosine_with_min_lr, min 0.1, warmup 3% | same | — |
| optimizer | AdamW (torch), β 0.9/0.999, ε 1e-8, wd 0 | same defaults | — |
| **numerics** | DeepSpeed ZeRO-3, bf16 mixed precision → **fp32 master weights and fp32 Adam moments** | DDP, model loaded in bf16 → **bf16 weights, bf16 Adam moments, no master copy** (8 bytes/param, measured) | bf16 has ~3 significant digits; weight updates of lr × grad can round to zero against bf16 weights. Over 3,000 steps this is a plausible contributor to "good teacher-forced loss, degenerate sampling" |
| Liger kernel | **off** | on, with `fused_linear_cross_entropy` | different loss kernel; FLCE produced NaN on an earlier stack of ours |
| attention | flash_attention_2 | sdpa (no aarch64 wheels for flash-attn) | numerically equivalent in practice |
| gradient checkpointing | on, `use_reentrant=False` | on, same default | — |
| sampling order | random (`group_by_length` False) | random | same (we changed this; our 4e-5 run used `group_by_length`) |
| hardware | 1 node × 8 H100 | 2 nodes × 4 GH200 | — |
| eval during training | none (`do_eval` False) | every epoch on a 1% held-out split | — |
| checkpoints | every N steps, keep 1 | every 604 steps, keep 6 | — |

## Data pipeline

| | theirs (TRL 0.16 `SFTTrainer`) | ours (`scripts/tokenize_dataset.py`) | effect |
|---|---|---|---|
| dataset | `open-r1/codeforces-cots` / `solutions_decontaminated` | same | — |
| rows trained on | **all 40,665** (3,170 steps × 128 / 10 epochs ≈ 40.6k per epoch) | **38,560**: drops 1,716 with `finish_reason="length"` (R1 truncated mid-trace), holds out 389 (1% of problems, by problem id) | they train on ~4% traces that never reach `</think>` or code; we don't |
| over-length traces | **head-truncated** at 32,768 tokens by TRL (`truncate_dataset`) — the tail, i.e. the code and the eos, is what gets cut | asserted ≤ 32,768; none exceeded after the finish_reason filter | — |
| **loss mask** | **none**: `DataCollatorForLanguageModeling(mlm=False)` → loss on every token, including the system prompt, the user prompt and `<\|im_start\|>assistant\n` | prompt masked to -100; loss only on the assistant turn (`labels = [-100]*prompt_len + input_ids[prompt_len:]`) | ~10% of their loss is on prompt tokens; different gradient, different "eval loss" meaning |
| chat template at training | Qwen2.5 stock template (`chat_template: None`) | same template | identical rendering of the conversation |
| `<think>` | inside the assistant content → supervised | same | — |
| sequence end | template emits `<\|im_end\|>\n`; TRL then **appends another `<\|im_end\|>`** because the last token is `\n`, not eos → `…<\|im_end\|>\n<\|im_end\|>` | `…<\|im_end\|>\n`, nothing appended | minor; they supervise two end tokens |
| pad token | `<\|endoftext\|>` at training time (Qwen default; TRL only overrides a missing pad). The **published** tokenizer has pad = eos = `<\|im_end\|>` — changed when the template was edited | `<\|endoftext\|>` | no training effect; matters for anyone fine-tuning *from* OlympicCoder |
| tokenizer identity | model's own | model's own; content-fingerprinted against the artifact | — |

## Chat template at inference

Their published `tokenizer_config.json` ends the generation prompt with
`<|im_start|>assistant\n<think>` — the model never has to produce the opening
`<think>` itself (blog, Lesson 4: without the prefill it "reverted to instruct-style
answers on out-of-domain prompts"). Ours ends at `<|im_start|>assistant\n`; the model
emits `<think>` on its own. Our generations do start with `<think>`, so this has not
been the failure, but it is a difference in what the model is asked to do.

## Evaluation

| | theirs (lighteval, `extended\|lcb:codegeneration`) | ours (`scripts/eval_lcb.py` on `lcb_runner`) | effect |
|---|---|---|---|
| problem set | `livecodebench/code_generation_lite` **v4_v5**, 268 problems | release_v6, 2024-08-01..15, **27 problems** | one problem = 3.7 points for us; their resolution is 10× finer |
| samples per problem | **16** (`codegen_pass@1:16`) | **1** | our pass@1 is one Bernoulli draw per problem |
| prompt | lighteval's generic LCB prompt: "You will be given a question (problem specification) and will generate a correct Python program…", starter-code branch | our SFT wrapper (`src/lcb_eval_prompts.py`): the long Python head from `solutions_py_decontaminated`, or the C++ head | theirs is the prompt a C++-trained model was never shown either — and it scored; ours tries to match training and the C++ variant did not help termination |
| chat template | `--use-chat-template` with the published template → **`<think>` prefilled** | no prefill | — |
| `max_new_tokens` | 32,768 | 30,720 (vllm rejects prompt + max_tokens > 32,768) | — |
| temperature / top_p | 0.6 / 0.95 | 0.6 / 0.95 | same |
| **stop sequences** | **none** (`stop_sequence=[]`, EOS only) | **`["###"]`** — `lcb_runner`'s parser default, passed to vLLM by `VLLMRunner`, never overridden by our argv | **22–23% of codeforces-cots traces contain `### Approach` / `### Solution Code` right after `</think>`**, i.e. where the answer starts. A model that learned that format is cut off at the heading, before its code block. Measured, not inferred: `openr1-distill-7b` closes `</think>` in 24/27 generations and **9 of those 24 end with no code after it** — a third of its answers discarded, and it still scores 40.7%. (The output never contains `###` because vLLM strips the stop string; absence proves nothing.) |
| code extraction | lighteval's `extract_code`: text between the **last two** ``` lines | `lcb_runner`'s `extract_code`: identical logic | **same** — not a difference |
| harness | vLLM via lighteval | vLLM 0.10.2 via `lcb_runner`, in a separate pixi env | — |
| scoring the C++ model | out-of-domain by their own note ("expects models to output solutions in Python") — and still scored | same mismatch; our C++-prompt run exists only to measure termination | — |

## Software

| | theirs | ours |
|---|---|---|
| transformers | 4.49.0 (model card) / 4.52.3 (setup.py) | 5.18.0 |
| trl | 0.16-era `SFTTrainer` (no `assistant_only_loss`/`completion_only_loss` fields in the pickled args) | none — plain `transformers.Trainer` |
| torch | 2.6.0 | 2.13.0+cu130 |
| deepspeed | 0.16.8 | none |
| liger-kernel | not used | 0.8.4 |
| accelerate | 1.4.0 | — |

## What this means, ranked

1. **The run in progress (job 24931873, resumed) is not the OlympicCoder recipe.**
   It uses lr 1e-5 and clip 0.2 from a recipe file that the actual run contradicts on
   both. Relative to the published training, its early updates are ~20× smaller and
   its later ones ~4× smaller. Whatever it produces will not be a reproduction.
2. **Our eval cuts answers before the code at the `###` heading** the training data
   taught the model to write. It affects every model trained on codeforces-cots —
   including `openr1-distill-7b` (9/24 completed answers lost; 40.7% is what
   survived, so its true pass@1 on this window is more like 55–60%). It does not
   explain the 24/27 generations of ours that never close `</think>` — those are
   degenerate (repeating tokens, 0 at the cap) — but it means every score we have
   is a floor. Fix: pass a `--stop` that never occurs, or patch `args.stop = []`
   before `VLLMRunner` reads it, then re-run everything.
3. **Numerics: fp32 master weights vs pure bf16.** DeepSpeed ZeRO-3 keeps fp32 master
   weights and optimizer states; our DDP run holds everything in bf16. This is the
   most plausible mechanism found so far for a model whose teacher-forced loss looks
   healthy while free-running generation degrades across epochs 8→10 — bf16 rounding
   turns small updates into noise, and 10 epochs is a lot of small updates. Untested.
4. **Loss on all tokens vs assistant-only.** Theirs also trains on the user prompt.
   Direction of effect unknown; it is a real difference in the objective.
5. **Statistical power.** 27 problems × 1 sample against 268 × 16. Our numbers cannot
   resolve differences smaller than ~10 points.
6. Smaller: `<think>` prefill at inference, the extra trailing `<|im_end|>`, FA2 vs
   SDPA, Liger FLCE on vs off, 4% R1-truncated traces kept vs dropped.

## What matches

Base model, dataset config, 10 epochs, effective batch 128, cosine-with-min-lr 0.1,
3% warmup, AdamW defaults, weight decay 0, bf16 compute, gradient checkpointing with
non-reentrant checkpoints, `packing=False`, max length 32,768, seed 42, random sample
order, temperature 0.6 / top_p 0.95 at eval, and a grad norm of ~2 at step 1 — which
says the two setups start from the same place before the differences above act.

## Appendix: second pass (2026-10-23)

Found by diffing every field of the pickled `training_args.bin` against defaults, and by
re-reading all synced generations with the stop-sequence finding in mind.

### Training arguments not covered above

| | theirs | ours | effect |
|---|---|---|---|
| `average_tokens_across_devices` | **False** | **True** (transformers 5 default) | theirs: each of the 8 devices normalises the loss by its own token count, then gradients are averaged — a device holding short sequences weighs as much as one holding long ones. Ours: normalised by the global token count, so every token weighs the same. Different objective under length variance this large (p10 ~1.3k, p90 ~23k tokens) |
| `optim` | `adamw_torch` | `adamw_torch_fused` (transformers 5 default) | same math, fused kernel; negligible |
| `dataloader_drop_last` | effectively drops the partial batch: 317 steps/epoch for 40,665 rows | keeps it: 302 steps/epoch for 38,560 | — |
| `use_seedable_sampler` | True | True | same |
| `warmup_ratio` → steps | 0.03 × 3,170 = 95 | 0.03 × 3,020 = 91 | same |
| `include_num_input_tokens_seen` | False | `"non_padding"` | logging only |
| `model_init_kwargs.use_cache` | False during training | transformers 5 handles it | — |
| `save_safetensors` | True | True | — |
| `zero3_save_16bit_model` | True — the pushed weights are the bf16 shard-gathered copy, not the fp32 master | n/a | the published checkpoint is bf16 like ours; the fp32 master existed only during training |

### Evaluation arguments not covered above

| | theirs | ours | effect |
|---|---|---|---|
| lighteval stop handling | `stop_sequence=[]` from the task, then `+ [eos_token]`; for chat models `stop_tokens = []` | `["###"]` | as above |
| `generation_config.json` of the evaluated model | OlympicCoder-7B ships `repetition_penalty 1.1, top_k 20, top_p 0.8, temperature 0.7` (Qwen defaults). lighteval builds `SamplingParams` from its own `generation_parameters` only, so **none of that is used** | `VLLMRunner` builds `SamplingParams` explicitly too → `repetition_penalty 1.0`, `top_k -1` | **same** — but anyone evaluating with HF `generate()` gets the 1.1 repetition penalty, which would suppress exactly the loops we see |
| `<think>` prefill | OlympicCoder-7B's template prefills; **OpenR1-Distill-7B's does not** (`chat_template.jinja` ends at `<|im_start|>assistant\n`) | ours do not prefill | **not the differentiator**: every one of our models opens `<think>` itself in 27/27 generations (26/27 for lr1e5-604), as does OpenR1-Distill-7B under our eval |

### What the new checkpoints say

| run | lr | epochs evaluated | `</think>` closed / 27 | pass@1 |
|---|---|---|---|---|
| 24691030 | 4e-5, clip 0.2, Liger FLCE, `group_by_length` | 8, 9, 10 | 3, 3, 3 | 0.037 each |
| 24931873 | 1e-5, clip 0.2, Liger FLCE, random | 2, 4, 6 | 4, 5, 3 | 0.037 each |
| `openr1-distill-7b` (reference) | — | — | 24 | 0.407 |

**Learning rate is ruled out as the cause of non-termination.** A 4× change in lr,
with everything else held, produced the same 3–5/27 termination rate from epoch 2
onward. The degeneration appears early (epoch 2, step 604) and does not move with
either lr or epoch count.

### Revised ranking of causes still standing

Everything that differs between our pipeline and a model that terminates (OpenR1-Distill-7B,
trained on the same codeforces-cots code split via TRL + DeepSpeed), and has not been
tested:

1. **Pure-bf16 weights and optimizer state vs fp32 master under ZeRO-3.** Untested, fits
   the symptom (teacher-forced loss fine, sampling degrades), and is the largest numerical
   difference left. Testable in one run: load the model in fp32 (`dtype=torch.float32`
   in `build_model`) with `bf16=True` autocast — memory goes 8 → 16 bytes/param, ~120 GB
   persistent on 7B, so it needs FSDP or ZeRO-3 on 4 GPUs. Or test at 0.5B first, where
   it fits trivially.
2. **Loss on prompt tokens vs assistant-only masking.** Theirs trains the prompt; ours
   masks it. One line to flip in `tokenize()` (`labels = input_ids`). Cheap to test.
3. **Liger `fused_linear_cross_entropy`.** Off for them, on for us, and it has a NaN
   history on our stack. One config flag; costs ~20 GB of memory at 7B (fits at 82 GB
   high-water — barely).
4. **`average_tokens_across_devices`.** One argument to match theirs.
5. **R1-truncated traces kept vs dropped, head-truncation of over-length traces.** They
   trained on ~4% of traces that never terminate; we dropped them. Direction is against
   this being the cause.
6. **The extra `<|im_end|>` TRL appends.** Their model saw `<|im_end|>\n<|im_end|>` at
   every sequence end; ours saw `<|im_end|>\n`. Cheap to replicate; unlikely to matter
   since both supervise the first `<|im_end|>`.

Not a difference, confirmed this pass: code extractor, repetition penalty/top-k at eval,
`<think>` opener, chat template at training time, optimizer math, warmup length,
sampler seeding.

## Appendix: third pass (2026-10-23)

### Tokenization, verified byte-for-byte

Reproduced TRL 0.16's path (`apply_chat_template(messages, tokenize=False)` → `tokenizer(text)` →
append eos if the last token is not eos) on a real `solutions_decontaminated` row and diffed it
against `scripts/tokenize_dataset.py`:

- rendered text: **identical**; `add_special_tokens` True vs False: **identical** (Qwen adds no BOS)
- **`theirs == ours + [<|im_end|>]`**, exactly — their only extra token is the appended eos after the
  template's trailing `\n`
- `<think>` is three BPE tokens (`<th`, `ink`, `>`) for both; neither side has a special token for it
- the prompt we mask and they train on is **11–26% of the sequence** (454–890 tokens of 2.5k–4.2k),
  not the ~10% guessed earlier. Their objective puts a quarter of its tokens, on short traces, on
  text we never score.

### Where the two runs are fp32 and bf16

| | theirs (DeepSpeed ZeRO-3, bf16) | ours (DDP, model loaded in bf16) |
|---|---|---|
| weights the optimizer updates | **fp32** master partition (`fp32_partitioned_groups_flat`) | bf16 |
| Adam `m`, `v` | **fp32** | bf16 |
| gradient accumulation across the 8 (ours: 16) micro-batches | **fp32** (`gradient_accumulation_dtype=torch.float32`, ZeRO-3 default) | bf16 `.grad` buffers |
| forward/backward compute | bf16 | bf16 |
| published weights | bf16 (`zero3_save_16bit_model`) | bf16 |

Three accumulations that are exact for them and rounded for us, over ~3,000 steps.

**How much is lost.** bf16 keeps 8 significand bits, so the spacing between representable values
near a weight `w` is `2^(⌊log2|w|⌋−7)`. On Qwen2.5-Coder-0.5B-Instruct (same family and scale as
7B) the 2-D weights have `|w|` p10/p50/p90 = 2.2e-3 / 1.2e-2 / 3.2e-2; the ulp at the median is
**6.1e-5**. An Adam step moves a weight by ≈ lr (m/√v is O(1)). An update smaller than half an ulp
rounds to nothing, and in pure bf16 nothing accumulates it for next time:

| peak lr | weights whose per-step update is below half a bf16 ulp |
|---|---|
| 4e-5 | **39.5%** |
| 1e-5 | **82.4%** |

And the cosine schedule spends most of the run below the peak. This is the standard failure mode
of bf16 training without an fp32 master copy (or stochastic rounding), and it fits every
observation at once: teacher-forced loss looks fine (the weights that do move are the high-gradient
ones, which are also the ones the loss is most sensitive to), free-running generation degrades, and
**lowering the learning rate makes it worse** — which is what the 1e-5 run showed. For them, with
fp32 master weights, 1e-5 vs 4e-5 would be an ordinary hyperparameter choice.

Caveat: `m/√v` is not exactly 1, early steps are larger, and this is a 0.5B weight histogram
standing in for 7B's. The fractions are indicative; the mechanism is not in doubt.

### A degeneration signature present in all of our runs and none of theirs

A digit `0` glued to an English word (`check0`, `code0`, `first0`, `the0`), with identifiers like
`x0`/`dp0` excluded:

| model | per 10k chars | generations affected |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct (base) | 0.00 | 0/27 |
| Qwen2.5-Math-7B-RoPE-300k (base) | 0.00 | 0/27 |
| `openr1-distill-7b` | 0.01 | 1/27 |
| `coding-distill-24321228` (Coder base, 4e-5, Python data) | 2.77 | 13/27 |
| `sft-math7b` (Math base, 4e-5, Python data) | 8.80 | 8/27 |
| `olympic-7b` epochs 8 / 9 / 10 (4e-5, C++) | 0.19 / 0.37 / 0.36 | 10 / 9 / 12 |
| `olympic-lr1e5` epochs 2 / 4 / 6 (1e-5, C++) | 0.03 / 0.02 / 0.32 | 1 / 2 / 3 |

It appears in every run from our pipeline — two bases, two datasets, two learning rates, two
sampling strategies — grows with training inside each run, and is absent from the bases and from
the model trained with TRL + ZeRO-3 on the same code data. Whatever causes it is in what all our
runs share: pure-bf16 optimization, Liger (with FLCE), assistant-only masking, transformers 5, DDP.

### Liger stays on

Turning Liger off is not an option on this hardware: without `fused_linear_cross_entropy` the fp32
logits alone are 32,768 × 152,064 × 4 B ≈ 20 GB plus their gradient, against an 82 GB high-water
mark on 96 GB cards. So the Liger difference cannot be isolated by ablation at 7B. It could be
isolated at 0.5B, where everything fits either way.

### Revised test plan

The fp32-vs-bf16 gap is the only remaining difference with a quantified mechanism, and it can be
closed without the memory of a full fp32 master copy:

1. **Stochastic-rounding or Kahan-compensated bf16 AdamW** — e.g. `torchao.optim.AdamW(...,
   bf16_stochastic_round=True)` or `optimi.AdamW(kahan_sum=True)`. Same 8 bytes/param as now,
   recovers the lost updates in expectation. One `optim` swap in `src/trainer.py`.
2. **fp32 master weights the way they had them** — needs 16 bytes/param ≈ 122 GB persistent at 7B,
   so FSDP or ZeRO-3 on 4 GPUs. The correct reproduction, at the cost of the sharding work that
   FLCE let us skip.
3. **Decide at 0.5B first**: pure bf16 vs fp32 master, same data, one epoch each, compare
   `pct_think_closed` and the glued-`0` rate. Hours, not days, and it settles whether (1) or (2) is
   worth the 7B run.

Independent of all of the above, and before any of these runs is scored: remove the `###` stop
sequence from `scripts/eval_lcb.py`, then re-score every existing checkpoint — the current numbers
are floors for every model including the reference.
