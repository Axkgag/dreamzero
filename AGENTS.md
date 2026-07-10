# Repository Guidelines

## Project Structure & Module Organization

DreamZero is a Python 3.11 research package. The installable source tree lives in `groot/`, with VLA data, model, experiment, config, and utility modules under `groot/vla/`. Training and conversion entry points live in `scripts/train/` and `scripts/data/`; inference helpers are in `scripts/inference/` and `eval_utils/`. Hydra YAML configs are stored under `groot/vla/configs/`. Project documentation is in `docs/`. `sim-evals/` is a separate uv-managed evaluation project with its own `pyproject.toml`, lockfile, and `src/sim_evals/` package.

## Build, Test, and Development Commands

- `pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129`: install the main package and CUDA PyTorch dependencies.
- `pip install -e ".[dev]"`: install development tools such as `pytest`, `black`, and `isort`.
- `python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path <checkpoint>`: run the distributed inference server.
- `python test_client_AR.py --port 5000`: smoke-test a running inference server.
- `bash scripts/train/droid_training_full_finetune.sh`: launch DROID full fine-tuning; inspect related scripts for LoRA, WAN2.2, AgiBot, and YAM variants.
- `cd sim-evals && uv sync && python run_eval.py --episodes <n> --scene <n> --headless`: set up and run simulation evaluation.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and descriptive `snake_case` names for modules, functions, variables, and config keys. Class names should use `PascalCase`. Keep configuration changes in YAML files near the relevant model or dataset config. Format Python with `black` and sort imports with `isort`; avoid broad refactors when changing research code paths.

## Testing Guidelines

There is no large committed test suite, so prioritize focused smoke tests. For server changes, run the distributed server plus `test_client_AR.py`. For dataset or config changes, run the smallest affected training, conversion, or evaluation command before scaling to multi-GPU jobs. Add new tests as `test_*.py` when behavior can be validated without heavyweight checkpoints or assets.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive commit subjects, for example `increase cache`, `clean wan5b backbone`, and `fix learning rate of lora training`. Keep commits scoped to one concern. Pull requests should summarize the motivation, list commands run, note required checkpoints/assets or environment variables, and include screenshots or logs when changing generated videos, simulation evaluation, or inference behavior.

## Security & Configuration Tips

Do not commit checkpoints, datasets, generated runs, Hugging Face tokens, API hosts, or private credentials. Keep large local assets under ignored directories such as `checkpoints/`, `data/`, `assets/`, or run output folders.
