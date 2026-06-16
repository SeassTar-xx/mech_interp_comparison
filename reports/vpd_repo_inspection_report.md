# VPD Repo Inspection Report

Date: 2026-06-16T18:33:16

## Repository

- Source URL: `https://github.com/goodfire-ai/param-decomp`
- Local path: `/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp`
- Commit: `1130c1fe66e7d526aadd169c2d2293bfa374863a`
- License: `MIT License`
- README canonical run: `goodfire/spd/runs/s-55ea3f9b`
- Core package: `param-decomp`, import name `param_decomp`
- Lab package: `param-decomp-lab`, import name `param_decomp_lab`
- Python requirement: core `==3.13.*`, lab `==3.13.*`

## Install Notes From README

README lists:

```bash
make install-dev
make install
make install-lab
```

The lab README lists CLI entrypoints including `pd-tms`, `pd-resid-mlp`, `pd-lm`, `pd-pretrain`, `pd-harvest`, `pd-autointerp`, and related postprocessing tools.

## Canonical Run and Artifacts

- VPD paper canonical 4L-pile run: `goodfire/spd/runs/s-55ea3f9b`.
- Repo config referencing it: `param_decomp_lab/clustering/configs/crc/pile_llama_simple_mlp-4L.json`.
- Runtime loader: `SavedLMRun.from_path(...)` in `param_decomp_lab/experiments/lm/run.py`.
- File resolver: `resolve_run_files(...)` in `param_decomp_lab/infra/run_files.py`.
- Expected local run files: `experiment_config.yaml` plus a latest `model_*.pth` checkpoint.
- If passed a W&B path and local cache is missing, `resolve_run_files` calls W&B APIs and downloads config/checkpoint files.
- Cache base is `PARAM_DECOMP_OUT_DIR`; this must be set to a project path before any real loader call.

## Artifact / Checkpoint Loading Paths

- `SavedLMRun.from_path(path)` resolves local directory or W&B run, then `load_model()` builds target model and loads the component model checkpoint.
- `component_model_io.load_component_model(...)` loads `model_*.pth` with `torch.load(..., weights_only=True)`.
- W&B helper functions live in `param_decomp_lab/infra/wandb.py` and `param_decomp_lab/infra/run_files.py`.
- App/backend route `param_decomp_lab/app/backend/routers/runs.py` also calls `SavedLMRun.from_path` and `load_model`.

## Visualization / Postprocessing

- App: `param_decomp_lab/app` with FastAPI backend and Svelte frontend.
- Harvest: `pd-harvest`, stores outputs under `PARAM_DECOMP_OUT_DIR/runs/<run_id>/harvest`.
- Autointerp: `pd-autointerp`, stores outputs under `PARAM_DECOMP_OUT_DIR/runs/<run_id>/autointerp`.
- Dataset attributions: `pd-attributions`.
- Graph interpretation: `pd-graph-interp`, stores graph interpretation outputs under `PARAM_DECOMP_OUT_DIR/runs/<run_id>/graph_interp`.
- Clustering pipeline: `pd-clustering`, `pd-cluster-harvest`, `pd-cluster-merge`.

## Commands That May Trigger Training Or Large Jobs

Do not run these in day-1 smoke testing without explicit confirmation:

- `pd-lm param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml`: fresh LM parameter decomposition training.
- `pd-lm --resume ...`: resumes training from `training_*.pth`.
- `pd-lm-layerwise ...`: submits/runs multiple LM decomposition jobs.
- `pd-pretrain ...`: trains target language models.
- `pd-tms ...` and `pd-resid-mlp ...`: train toy/model decomposition experiments.
- `pd-harvest`, `pd-attributions`, `pd-autointerp`, `pd-graph-interp`, `pd-postprocess`: not training the decomposition itself, but can trigger large postprocessing/SLURM jobs and W&B/artifact access.

## Static Examples Found

- Pronoun/circuit examples are described in `papers/Interpreting_Language_Model_Parameters.md`, including prompts like `The princess lost her crown.` and `The prince lost his crown.`
- Bracket/delimiter example described in the paper: `< u , v >` predicting `>`.
- Editing README demonstrates `EditableModel.from_wandb("goodfire/spd/s-892f140b")`, but that is a different run from the canonical 4L-pile run.

## External Access Check

```text
# VPD external access checks
2026-06-16T18:28:02+08:00
WANDB_API_KEY_set=no
curl_run_page_status
HTTP/1.1 200 Connection established

HTTP/2 200 
x-powered-by: Express
set-cookie: host_session_id=00e4694f-1d31-4e68-8fd9-316f1fb34a92; Max-Age=31536000; Domain=wandb.ai; Path=/; Expires=Wed, 16 Jun 2027 10:28:02 GMT; Secure; SameSite=None
document-policy: js-profiling
content-type: text/html; charset=utf-8
content-length: 3856
etag: W/"f10-EczoJ9Sz363nQ9U8ZfFB1KY5oUI"
date: Tue, 16 Jun 2026 10:28:02 GMT
via: 1.1 google
alt-svc: h3=":443"; ma=2592000,h3-29=":443"; ma=2592000

curl_api_files_status
HTTP/1.1 200 Connection established

HTTP/2 405 
vary: Origin
x-wandb-trace-id: c5bfbd85a27f04ee6a1fd72051bad8af
date: Tue, 16 Jun 2026 10:28:03 GMT
via: 1.1 google
alt-svc: h3=":443"; ma=2592000

curl_api_files_get_prefix
HTTP/1.1 200 Connection established

HTTP/2 404 
content-type: application/json; charset=utf-8
vary: Origin
x-content-type-options: nosniff
x-wandb-trace-id: b08ed4acc25c369bc7628fabd99883c1
date: Tue, 16 Jun 2026 10:28:22 GMT
content-length: 69
via: 1.1 google
alt-svc: h3=":443"; ma=2592000,h3-29=":443"; ma=2592000

body_bytes=69
{"error":"file h7nbs2rt/s-55ea3f9b not found while downloading file"}

```

The W&B run page returned HTTP 200 without a local API key, but the repository loader would still need W&B SDK access and would download checkpoint files. No checkpoint download was attempted.
