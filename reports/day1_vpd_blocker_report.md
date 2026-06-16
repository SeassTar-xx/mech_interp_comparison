# Day 1 VPD Blocker Report

Date: 2026-06-16T18:33:16

## Summary

VPD code inspection succeeded, but executable artifact reproduction is blocked today. The main blockers are: no local VPD checkpoint/artifact found, current Python is 3.10 while the repo requires Python 3.13, and canonical run loading would require W&B SDK access plus checkpoint download.

## Repo Status

- Repo cloned: `True`
- Repo URL: `https://github.com/goodfire-ai/param-decomp`
- Commit: `1130c1fe66e7d526aadd169c2d2293bfa374863a`
- License: `MIT License`

## Install Status

Safe install check used `pip install --dry-run --target ... -e repos/param-decomp` and did not install the package.

```text
Obtaining file:///mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Getting requirements to build editable: started
  Getting requirements to build editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
INFO: pip is looking at multiple versions of param-decomp to determine which version is compatible with other requirements. This could take a while.
ERROR: Package 'param-decomp' requires a different Python: 3.10.20 not in '==3.13.*'

```

Python 3.13 availability check:

```text
# Python 3.13 availability check
2026-06-16T18:31:16+08:00
python3.13: not found
python3.13m: not found
uv: not found
conda: not found

```

Import smoke test:

```text
python 3.10.20 | packaged by conda-forge | (main, Mar  5 2026, 16:42:22) [GCC 14.3.0]
PARAM_DECOMP_OUT_DIR /mnt/20t/xuxin/mech_interp_comparison/outputs/vpd/param_decomp_out
param_decomp OK /mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/param_decomp/__init__.py
param_decomp_lab OK /mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/param_decomp_lab/__init__.py
param_decomp.component_model FAIL ImportError cannot import name 'override' from 'typing' (/mnt/8t/xhm/miniconda3/envs/EasyEdit1/lib/python3.10/typing.py)
param_decomp_lab.infra.wandb FAIL SyntaxError invalid syntax (wandb.py, line 235)
param_decomp_lab.experiments.lm.run FAIL ModuleNotFoundError No module named 'fire'

```

## Canonical Run Access

- Canonical run from README: `goodfire/spd/runs/s-55ea3f9b`.
- Browser-style W&B page request returned HTTP 200, suggesting the page is externally reachable.
- No `WANDB_API_KEY` is set in the checked environment.
- W&B SDK is not installed in the current env, and the repo cannot be installed under Python 3.10.
- Calling `SavedLMRun.from_path("goodfire/spd/runs/s-55ea3f9b")` was intentionally not attempted because code inspection shows it can download missing config/checkpoint files.

## Local Artifact Status

- No obvious local VPD/Goodfire checkpoint or canonical run directory was found in `/mnt/20t/xuhaoming/models/` or the project root.
- Local candidates after clone are mostly repo configs/paper text, not completed run artifacts:

```text
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/typings
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/.mcp.json
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/LICENSE
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/conftest.py
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/uv.lock
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/param_decomp.egg-info
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/.pre-commit-config.yaml
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/Makefile
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/CLAUDE.md
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/.gitignore
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/nano_param_decomp
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/MANIFEST.in
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/param_decomp_lab
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/.vscode
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/README.md
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/.env.example
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/param_decomp
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/pyproject.toml
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/.github
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/typings/transformers
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/typings/orjson
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/typings/transformers/__init__.pyi
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/typings/orjson/__init__.pyi
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Interpreting_Language_Model_Parameters.md
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Stochastic_Parameter_Decomposition
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/apd_paper.md
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/feature_response_with_subnets_42_1layers_8qz1si1l.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/subnetworks_graph_plots.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_feature_response_single_2layers.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/model_functions_paper.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_feature_response_single_1layers.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_avg_components_scatter_1layers_8qz1si1l.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/minimality_target-1.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/tms_combined_diagram.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_per_feature_performance_1layers_9a639c6w.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_weights_1layers_8qz1si1l.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_per_feature_performance_1layers_8qz1si1l.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/feature_response_with_subnets_42_2layers_cb0ej7hj.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_scrub_hist_2layers_cb0ej7hj.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_avg_components_scatter_2layers_cb0ej7hj.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/apd_tms_overview-1.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid2_side-1.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/l_p_formulation-1.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid1_side-1.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_scrub_hist_1layers_8qz1si1l.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_per_feature_performance_2layers_wbeghftm.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_per_feature_performance_2layers_cb0ej7hj.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_feature_response_multi_1layers.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_weights_2layers_cb0ej7hj.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/spd_target2-1.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Attribution_based_Parameter_Decomposition/figures/resid_mlp_feature_response_multi_2layers.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Stochastic_Parameter_Decomposition/figures
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Stochastic_Parameter_Decomposition/spd_paper.md
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Stochastic_Parameter_Decomposition/figures/tms_vectors_5-2-identity.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Stochastic_Parameter_Decomposition/figures/neuron_contribution_pairs_2layers_wau744ht.png
/mnt/20t/xuxin/mech_interp_comparison/repos/param-decomp/papers/Stochastic_Parameter_Decomposition/figures/resid_mlp_weights_1layers_ziro93xq.png
```

## Commands Not Run

The following are blocked because they train, resume training, submit jobs, or trigger large artifact workflows:

- `make install-dev`: installs full dev/lab dependencies and hooks, requires Python 3.13.
- `pd-lm ...`: trains LM parameter decomposition.
- `pd-lm --resume ...`: resumes training.
- `pd-lm-layerwise ...`: submits/runs multiple decomposition jobs.
- `pd-pretrain ...`: trains target language models.
- `pd-harvest`, `pd-attributions`, `pd-autointerp`, `pd-graph-interp`, `pd-postprocess`: may launch large postprocessing/SLURM/W&B workflows.
- `SavedLMRun.from_path("goodfire/spd/runs/s-55ea3f9b")`: may download W&B run files/checkpoints because local cache is absent.

## Questions For Advisor / Team

- Is `goodfire/spd/runs/s-55ea3f9b` intended to be public through W&B API, or do we need a Goodfire/W&B account with access?
- What is the expected checkpoint size for the canonical 4L-pile VPD run?
- Can they provide a tarball or local mirror of `experiment_config.yaml`, latest `model_*.pth`, and any harvest/autointerp/graph outputs?
- Which run should be treated as the canonical comparison entry: `s-55ea3f9b`, `s-892f140b`, or another paper-release run?
- Is Python 3.13 acceptable to create under `/mnt/20t/xuxin/mech_interp_comparison/env/`, or should we use a provided environment?
