Push-F minimal bundle

This has the small set of files someone will need to collect data (optional) and run training.

What's here
- scripts/collect_pushf_dataset.py  (non-interactive by default)
- config/train_pushf_remote.yaml    (Hydra config tuned for remote/GPU)
- conda_environment.yaml            (env spec)

Quick start

1) Unzip or clone this folder on the remote machine and cd into it.

2) Make the conda env and activate it:

```bash
conda env create -f conda_environment.yaml -n robodiff
conda activate robodiff
```

3) If you want the remote to collect the dataset (no human input required):

```bash
# from bundle root
python scripts/collect_pushf_dataset.py --out data/pushf/pushf_replay.zarr --episodes 200 --steps 300 --no-render --agent scripted
```

4) Run training (point to the dataset path):

```bash
python train.py +diffusion_policy/config/train_pushf_remote.yaml task=pushf_image task.dataset.zarr_path=/absolute/path/to/data/pushf/pushf_replay.zarr
```

Notes
- This bundle doesn't include the full `diffusion_policy/` package. If the trainer doesn't have the repo, ask them to clone https://github.com/skandima/diffusion_policy_pushf and checkout `pushf/remote`.

