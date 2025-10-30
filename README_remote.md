Push-F minimal package — instructions for remote trainer

This bundle contains the minimal files needed for someone else to collect data (if needed) and run training for the Push-F task.

Files included:
- scripts/collect_pushf_dataset.py  # non-interactive collection script (default: scripted)
- config/train_pushf_remote.yaml    # remote-friendly training config (Hydra)
- conda_environment.yaml            # conda env spec (robodiff)

Quick steps for the remote trainer

1) (Optional) If you received this as a zip: unzip into a directory and cd into it.

2) Create and activate conda env:

```bash
conda env create -f conda_environment.yaml -n robodiff
conda activate robodiff
```

3) If you will collect data on the remote machine, run the non-interactive collector:

```bash
# from bundle root
python scripts/collect_pushf_dataset.py --out data/pushf/pushf_replay.zarr --episodes 200 --steps 300 --no-render --agent scripted
```

This creates `data/pushf/pushf_replay.zarr` inside the bundle.

4) To run training (if the full training code is available on the remote), point the training command at the dataset path and use the provided Hydra config. Example (replace with your training entrypoint if different):

```bash
# example: run training and override dataset path
python train.py +diffusion_policy/config/train_pushf_remote.yaml task=pushf_image task.dataset.zarr_path=/absolute/path/to/data/pushf/pushf_replay.zarr
```

Notes
- This bundle intentionally omits the full `diffusion_policy/` package to keep size small. If the remote user does not already have the full training code, they should clone the repo at https://github.com/skandima/diffusion_policy_pushf and use the `pushf/remote` branch.
- If you want me to include a small subset of `diffusion_policy/` needed for training so this bundle is self-contained, tell me and I will add it (I can identify which modules are required and include them).

Contact
- If anything fails, paste error outputs and I will help debug.
