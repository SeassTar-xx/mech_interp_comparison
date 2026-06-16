# Day-1 VPD / NLA Comparison Summary

Date: 2026-06-16

This is a feasibility and comparison-framework summary only. It should not be read as a scientific conclusion.

## 1. Task Understanding

I am currently responsible for two lines in the comparison project: VPD and NLA. The project rule is to avoid retraining, fine-tuning, or training a new explainer; the first-stage goal is to use existing checkpoints, released artifacts, and public repos to check feasibility and build a comparison entry.

NLA and VPD are not fully apples-to-apples because they explain different objects.

- NLA explains activation-level states, here the Qwen2.5-7B-Instruct layer-20 residual stream, and turns an activation vector into a natural-language explanation with AV. AR can map an explanation back into an activation vector for reconstruction checks.
- VPD explains parameter-level subcomponents in a trained model and can support mechanism evidence through causal importance, ablation effect, adversarial ablation, and attribution graph metadata.

For day 1, NLA is closer to same-base-model behavior analysis for safety/bias/sycophancy prompts. VPD is better treated as a canonical parameter-level case study first, because the available Goodfire artifact targets a small 4-layer Pile model rather than Qwen/Llama instruction behavior.

## 2. Method Comparison Table

| method | source | available model/checkpoint | interpreted object | output format | natural language output or not | causal intervention support | needs additional training or not | current artifact availability | suitable tasks | main risks |
|---|---|---|---|---|---|---|---|---|---|---|
| NLA | `kitft/nla-inference`; checkpoints `kitft/nla-qwen2.5-7b-L20-av` and `kitft/nla-qwen2.5-7b-L20-ar` | Base model `/mnt/20t/xuhaoming/models/Qwen2.5-7B-Instruct`; AV/AR checkpoints under `/mnt/20t/xuhaoming/models/` | Layer-20 residual activation, last prompt token | AV natural-language explanation; AR reconstruction score | Yes | Indirect. AR gives reconstruction evidence, but AV text itself is not a causal intervention | No. Uses released AV/AR checkpoints | Available locally and smoke-tested; final SGLang AV run processed `16` prompts | Safety/refusal, bias/stereotype, sycophancy, evaluation awareness prompts on Qwen2.5 | Explanations are readable but can be non-causal or over-interpreted; text may describe generic Q&A patterns |
| VPD | `goodfire-ai/param-decomp`; paper tag `vpd-paper`; canonical W&B run `goodfire/spd/runs/s-55ea3f9b` | VPD checkpoint `model_400000.pth` and target pretrain checkpoint `model_step_99999.pt` downloaded under project outputs | Parameter subcomponents of a small `LlamaSimpleMLP` target model | Component weights, causal-importance outputs, ablation/graph evidence when extracted | Not inherently. Textual interpretation requires postprocessing/autointerp or human analysis | Yes. Designed around causal importance and ablation/adversarial ablation evidence | No. Released artifact loaded; no VPD training run | Available locally; `ComponentModel` load smoke test passed; pronoun next-token smoke test passed | Pronoun prediction, bracket/delimiter closing, paper/blog case studies | Small model does not directly represent Qwen/Llama instruction behavior; graph/cluster artifacts may be incomplete or internal |
| SAE | Placeholder, not current primary scope | Not checked in this day-1 run | Activation features | Sparse features and feature descriptions if available | Sometimes, if feature labels/autointerp exist | Possible through activation patching/ablation | Usually requires existing SAE checkpoint, not new training for this project | Not inspected | Future activation-level baseline | Requires matching model/layer SAE artifacts |
| ICA | Placeholder, not current primary scope | Not checked in this day-1 run | Activation directions/components | Independent components and scores | No, unless post-hoc labels are added | Possible through direction intervention | Could be fitted, but fitting is outside current no-training principle unless using existing artifacts | Not inspected | Lightweight baseline if existing components are available | May be unstable and less semantically clean |

## 3. Evaluation Framework

Following the spirit of “Prediction, Explanation, or Over-interpretation?”, the comparison should separate model behavior from explanation plausibility. The same prompt family should support five checks:

- Prediction consistency: whether the method's implied claim matches the model's actual output or token probabilities.
- Explanation consistency: whether repeated prompts or paraphrases produce stable explanation content.
- Cross-method consistency: whether different explanation methods identify compatible mechanisms or failure modes.
- Generalizability: whether explanations/components transfer across paraphrases, counterfactual prompts, and categories.
- Over-interpretation risk: whether a readable or appealing explanation is being mistaken for faithful mechanism evidence.

### NLA Evaluation

For NLA, prediction consistency should start by extracting concrete claims from the AV explanation, then comparing them with the base model answer, top tokens, and yes/no logprobs. In the final NLA run, processed categories were: `bias_stereotype: 4, evaluation_awareness: 3, safety_refusal: 5, sycophancy: 4`. Example: for `safety_refusal_001`, Qwen answered “No” with `No` as the top token, and the NLA explanation discussed a structured prohibition/negative-answer pattern.

Explanation consistency should check whether NLA mentions the intended prompt attributes: safety/refusal, bias/stereotype, sycophancy, or evaluation awareness. Generalizability should be tested with paraphrases, counterfactual pairs, and category-balanced prompts. The main risk is that NLA text is very readable but not automatically causal; it may also produce generic descriptions of answer format rather than the actual reason for the model behavior.

### VPD Evaluation

For VPD, prediction and explanation should first be evaluated on the small model's token-level behavior, not on Qwen/Llama instruction tasks. Suitable first cases are pronoun prediction and bracket/delimiter closing. The day-1 pronoun smoke test found:

- `The princess lost her crown.` top next tokens included '\n', ' She', ' The', '  ', ' Her'.
- `The prince lost his crown.` top next tokens included '\n', ' He', ' The', '  ', ' His'.

Faithfulness should then be evaluated using causal-importance outputs and ablation effects, not merely qualitative stories. Generalizability should test whether the same subcomponents recur across a prompt family. The main risk is scope mismatch: the available VPD artifact is a small-model canonical artifact, so it should not be claimed to explain Qwen2.5 safety/bias/sycophancy behavior unless suitable artifacts exist.

## 4. Recommended Next Experiments

### NLA

- Fix the first NLA setting to Qwen2.5-7B-Instruct layer 20.
- Expand mini prompts to safety, bias, and sycophancy, roughly 50 to 200 examples per category.
- Add an SR baseline on the exact same prompt set.
- Keep saving base answer, top tokens, yes/no logprobs when available, AV explanation, and AR reconstruction score.
- Add AR reconstruction score as a routine quantitative diagnostic rather than a separate optional check.

### VPD

- Keep the downloaded Goodfire canonical run as the first VPD comparison entry.
- Ask whether the group has an internal mirror of VPD checkpoint, cluster mapping, harvest, postprocess, or attribution graph artifacts.
- Prioritize reproducing one existing case study, such as pronoun prediction or bracket closing, rather than training a new decomposition.
- If additional artifacts are unavailable, present VPD as a method-level comparison entry and paper/blog-based case study, not as a Qwen safety/bias/sycophancy same-model benchmark.

## 5. Day-1 Status

### NLA Status

- Local base model found: `/mnt/20t/xuhaoming/models/Qwen2.5-7B-Instruct`.
- Local AV checkpoint found: `/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-av`.
- Local AR checkpoint found: `/mnt/20t/xuhaoming/models/nla-qwen2.5-7b-L20-ar`.
- Final status: `ok`.
- Final AV method: `sglang_input_embeds`.
- Final prompt count: `16`.
- SGLang status: `ok`; server stopped: `True`.
- AR available: `True`.
- Sample output exists in `outputs/nla/20260616_160604_sglang_av/nla_explanations.jsonl`.

### VPD Status

- Repo cloned and inspected: yes.
- Paper-tag worktree created: `/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp-vpd-paper`.
- Python 3.13 environment created: `True`.
- Canonical VPD checkpoint downloaded: `True`, size `2908595703` bytes.
- Target pretrain artifact downloaded: `True`, checkpoint size `267736421` bytes.
- Component model load: `True`.
- Pronoun prediction smoke test: `True`.
- Training run executed: `False`.

### Remaining Blockers

- NLA needs a larger, balanced prompt set and SR baseline before comparison claims.
- NLA explanations need claim extraction and consistency scoring to reduce over-interpretation.
- VPD still needs a focused extraction script for top causal components and ablation/graph evidence.
- VPD cluster mapping and postprocess artifacts may be Goodfire-internal and not fully mirrored in the public checkpoint.

## 6. Questions For Tomorrow's Meeting

- Does the group already have VPD artifact/checkpoint mirrors, especially cluster mappings, harvest outputs, or attribution graph files?
- Should NLA be fixed first to Qwen2.5-7B-Instruct layer 20 for all day-2 experiments?
- Should VPD be positioned as a parameter-level case study rather than a same-model safety/bias/sycophancy benchmark?
- How many NLA samples are expected for the first internal comparison table: 50, 100, or 200 per category?
- Should SR baseline be added immediately on the same NLA prompt set?
- For NLA, should we score explanations manually first, or build a simple claim-extraction template?
- For VPD, is one faithful canonical case study enough for the first comparison draft?

## Source Files Read

- `reports/day1_nla_smoke_test_report.md`
- `reports/day1_vpd_smoke_test_report.md`
- `reports/day1_vpd_blocker_report.md`
- `reports/nla_local_artifact_report.md`
- `reports/vpd_local_artifact_report.md`
- `reports/vpd_repo_inspection_report.md`
- `outputs/nla/20260616_160604_sglang_av/`
- `outputs/vpd/20260616_191600_smoke/`
