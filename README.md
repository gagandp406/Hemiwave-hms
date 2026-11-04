# HemiWave

**A lightweight hemisphere-symmetric waveform network for ensemble classification of harmful brain activity in EEG**

Gagandeep Singh, Iman Dehzangi — Department of Computer Science, Rutgers University, Camden, NJ, USA

This repository holds the code and result summaries for the HemiWave paper (submitted to *IEEE Journal of Biomedical and Health Informatics*). It classifies 50-s EEG segments into six classes (Seizure, LPD, GPD, LRDA, GRDA, Other) and predicts the distribution of expert votes.

> **Naming:** the model is called **HemiWave** in the paper and **PRISM** in the code (`prism.py`, `PRISM_EEG.ipynb`, run tags such as `prism_wf5`). They are the same model; the code names were kept so that archived results and run tags stay valid.

## Main results

Out-of-fold, patient-grouped 5-fold cross-validation on 6,350 segments with at least ten expert votes (1,063 patients). KLD = Kullback–Leibler divergence to the expert vote distribution (lower is better).

| Model | Parameters | KLD [95% CI] |
|---|---|---|
| Reference classifier, single instance | 12.9 M | 0.2217 [0.2114–0.2321] |
| HemiWave alone | **2.96 M** | 0.2732 [0.2588–0.2877] |
| Reference + HemiWave | 15.9 M | 0.2177 [0.2074–0.2277] |
| Reference ×3 | 38.8 M | 0.2145 [0.2047–0.2247] |
| Reference ×2 + HemiWave | 28.8 M | 0.2143 [0.2039–0.2248] |
| **Reference ×3 + HemiWave** | 41.8 M | **0.2131 [0.2032–0.2232]** |

- The best ensemble lowers KLD by 0.0086 relative to a single reference instance (95% CI −0.0116 to −0.0059).
- Two reference instances plus HemiWave match three reference instances with **26% fewer parameters**.
- Gains are largest on segments where experts disagree.

Full reports: [`results/comparisons/full_comparison/report.md`](results/comparisons/full_comparison/report.md) and [`results/comparisons/extended_eval/report.md`](results/comparisons/extended_eval/report.md).

## Method in brief

- **HemiWave** (`prism.py`): a shared 1-D residual encoder reads each of the 18 bipolar channels; learned autocorrelation tokens capture periodicity; homologous left/right channels are fused as (L+R)/2 and |L−R|, which makes the output exactly invariant to swapping the hemispheres; a 2-layer transformer models change over the 50-s window. 2.96 M parameters.
- **Reference classifier** (`VIPEEGNet_local.ipynb`): our re-implementation of the image-based classifier of Sun et al. (*npj Digit. Med.* 8:768, 2025), trained with three random seeds.
- **Ensemble** (`blend_5fold.py`, `full_comparison.ipynb`): reference instances are averaged, then blended with HemiWave by a weighted geometric mean. The single weight is chosen leave-one-fold-out, so it never sees the labels of the fold it is applied to.

## Repository layout

```
.
├── setup.sh, setup_torch_env.sh, setup_effnet_env.sh   environments (no sudo needed)
├── download_data.sh, requirements*.txt                 data download and Python packages
├── prism.py, prism_train.py, PRISM_EEG.ipynb           HemiWave model, training loop, run notebook
├── run_prism.sh                                        HemiWave runs (fold-0 screens, final 5-fold run)
├── VIPEEGNet_local.ipynb, run_vipeegnet.sh             reference classifier (3 seeds)
├── EffNetB0_local.ipynb, run_effnet.sh                 EfficientNet-B0 spectrogram baseline (supplementary)
├── blend_fold0.py, blend_5fold.py                      ensemble weight selection and blending
├── compare_models.ipynb, compare_all_models.ipynb      model comparisons
├── eval_top10.ipynb, eval_extended.ipynb               clinical and statistical evaluations
├── full_comparison.ipynb                               final comparison of all models and ensembles (paper numbers)
├── export_full_oof.py, load_results.py                 out-of-fold export and loader
├── archive_run.sh, archive_vipeegnet.sh                freeze a finished run (read-only, checksummed)
├── experiments/                                        exploratory code not used in the paper (CORAL, TTA, checks)
└── results/
    ├── comparisons/        reports, CSV tables and figures from each evaluation notebook
    ├── runs/<run>/         per-run summary, patient folds, training curves, environment
    └── executed_notebooks/ executed copies of the notebooks behind the reported numbers
```

## Environment and setup

### Hardware and system used

| Item | Used for the paper | Suggested minimum (estimate) |
|---|---|---|
| Machine | Dell Precision 5820 workstation | any Linux x86-64 machine |
| GPU | NVIDIA RTX A6000, 48 GB | NVIDIA GPU with 16 GB or more (smaller GPUs need a smaller batch size) |
| NVIDIA driver | 580 | 525 or newer (CUDA 12 wheels) |
| System memory | 256 GB | 64 GB |
| Operating system | Ubuntu 22.04 | any recent Linux |
| Python | system Python 3.10 | 3.10 or 3.11 |
| Disk space | about 120 GB | about 50 GB (20 GB download, extracted EEGs, EEG cache, outputs) |

No system CUDA toolkit and no `sudo` are needed: the GPU builds of TensorFlow and PyTorch bring their own CUDA and cuDNN libraries, and the setup scripts create the virtual environments without root.

### The three Python environments

Each model family has its own virtual environment and Jupyter kernel, because the reference model, HemiWave and the EfficientNet-B0 baseline need incompatible library versions. All three are created inside the project directory (`$HMS_PROJECT_DIR`, default `~/hms`).

| Environment | Created by | Jupyter kernel | Main packages | Used for |
|---|---|---|---|---|
| `.venv` | `setup.sh` | `hms-vipeegnet` | TensorFlow 2.19.0 `[and-cuda]` (Keras 3), NumPy, pandas, scikit-learn, papermill, Kaggle CLI | data download, reference classifier (3 seeds), EEG cache |
| `.venv-torch` | `setup_torch_env.sh` | `hms-torch` | PyTorch 2.5.1 (CUDA 12.4 wheels), timm 1.0.11, huggingface_hub 0.26.2, NumPy < 2, pandas, SciPy, scikit-learn, matplotlib, seaborn, papermill | HemiWave, ensemble blending, all evaluation notebooks |
| `.venv-tf215` | `setup_effnet_env.sh` | `hms-effnet` | TensorFlow 2.15.1 `[and-cuda]` (Keras 2), efficientnet 1.1.1, NumPy 1.26.4, pandas 2.2.3, librosa 0.10.1, setuptools < 81 | EfficientNet-B0 baseline (supplementary) |

The package lists are in `requirements.txt` (reference), `requirements_effnet.txt` (EfficientNet-B0) and `setup_torch_env.sh` (PyTorch). The exact versions installed for each archived run are recorded in `results/runs/<run>/env/pip_freeze.txt`, together with the GPU and driver details in `env/nvidia_smi.txt`.

Other one-time downloads made by the setup scripts:

- ImageNet weights for EfficientNet-B0 (AutoAugment version), saved to `$HMS_PROJECT_DIR/pretrained/` by `setup_effnet_env.sh`.
- ImageNet weights for timm's `efficientnet_b0`, cached by `setup_torch_env.sh`. These are used only by HemiWave's early image-map variants (Fig. 2, iteration 1), not by the final model.

### Data access

The dataset is downloaded with the Kaggle command-line tool, which is installed in `.venv`:

1. Create an API token (Kaggle → Settings → API → Create New Token) and save it as `~/.kaggle/kaggle.json`.
2. Run `chmod 600 ~/.kaggle/kaggle.json`. **Never commit this file**; `.gitignore` excludes it.
3. Accept the rules of the HMS Harmful Brain Activity Classification dataset on its page.
4. Run `./download_data.sh`. It downloads the 20 GB archive and extracts `train.csv`, `train_eegs/`, `test.csv` and `test_eegs/` into `$HMS_PROJECT_DIR/hms-data/`. Set `KEEP_SPEC=1` to also extract the spectrograms, which only the EfficientNet-B0 baseline needs.

### Directory layout at run time

Code stays in this repository; everything large is written to the project directory:

```
$HMS_PROJECT_DIR (default ~/hms)
├── .venv/, .venv-torch/, .venv-tf215/   Python environments
├── hms-data/                           raw data (download_data.sh)
├── eeg_cache_v2/                       18-channel bipolar, filtered EEGs, one .npy per recording (built by the reference notebook)
├── pretrained/                         ImageNet weights for the EfficientNet-B0 baseline
├── fair_comparison/<run>/              out-of-fold predictions of each run
├── results/<run>/                      archived runs (read-only, checksummed) and results/comparisons/
└── runs/                               executed copies of every notebook run
```

## Reproducing the results

The steps must run in this order: HemiWave reads the EEG cache built by the reference notebook and uses the reference model's patient folds, and the evaluation notebooks read the archived runs. Run the long jobs inside `tmux` so they survive a disconnect.

```bash
git clone <repo-url> ~/hemiwave-hms && cd ~/hemiwave-hms
export HMS_PROJECT_DIR=~/hms

# 1. Environments (each script checks that the GPU is visible) and data
./setup.sh                  # .venv, kernel hms-vipeegnet
./setup_torch_env.sh        # .venv-torch, kernel hms-torch
./setup_effnet_env.sh       # .venv-tf215, kernel hms-effnet (only for the EfficientNet-B0 baseline)
./download_data.sh

# 2. Put the notebooks and helpers where the run scripts expect them
cp *.ipynb *.py "$HMS_PROJECT_DIR"/
mkdir -p "$HMS_PROJECT_DIR/results" && cp load_results.py "$HMS_PROJECT_DIR/results/"

# 3. Reference classifier: seed 2024 (several hours; also builds eeg_cache_v2), then archive it
./run_vipeegnet.sh smoke    # optional 10-15 min check
./run_vipeegnet.sh full
./archive_vipeegnet.sh      # -> results/vipeegnet_groupkfold_v1 (patient folds used by every other model)

# 4. Two more reference seeds (several hours each), archived with archive_run.sh
./run_vipeegnet.sh seed 2025
./run_vipeegnet.sh seed 2026

# 5. HemiWave: fold-0 development runs (paper Fig. 2), then the final 5-fold model (~1.5 h)
./run_prism.sh screen && ./run_prism.sh screen3 && ./run_prism.sh screen4
./run_prism.sh full5
./archive_run.sh prism_wf5_v1 ~/hms/fair_comparison/prism_wf5 ~/hms/prism/output_prism_wf5 ~/hms/.venv-torch prism_wf5 PRISM_EEG.ipynb
./archive_run.sh prism_wf5_tta_v1 ~/hms/fair_comparison/prism_wf5_tta ~/hms/prism/output_prism_wf5 ~/hms/.venv-torch prism_wf5 PRISM_EEG.ipynb

# 6. Optional EfficientNet-B0 baseline
./run_effnet.sh native      # archive command: see the header of archive_run.sh

# 7. Ensemble and evaluation (PyTorch environment)
source "$HMS_PROJECT_DIR/.venv-torch/bin/activate"
python blend_5fold.py prism_wf5_tta blend_vip_prism_wf5
cd "$HMS_PROJECT_DIR"
papermill full_comparison.ipynb runs/full_comparison.ipynb -k hms-torch -p PROJECT_DIR "$HMS_PROJECT_DIR"
```

`eval_top10.ipynb`, `eval_extended.ipynb` and `compare_all_models.ipynb` run the same way. `full_comparison.ipynb` produces the numbers reported in the paper.

Approximate run times on the RTX A6000: reference model, several hours per seed; HemiWave, about 16 minutes per fold (1.5 h for five folds); each evaluation notebook, a few minutes.

## Data

The EEG recordings and expert votes come from the public Harmful Brain Activity Classification dataset collected at Massachusetts General Hospital (Jing et al., *Neurology* 100(17), 2023). The data are not included in this repository; `download_data.sh` fetches them. Recordings are converted to an 18-channel bipolar montage, clipped to ±1024 µV and band-pass filtered at 0.5–45 Hz.

## Acknowledgements

- The reference classifier re-implements the method of Sun et al., *npj Digital Medicine* 8:768 (2025).
- The EfficientNet-B0 baseline follows a public spectrogram starter notebook by C. Deotte.

## Citation

```
@article{singh2026hemiwave,
  title   = {HemiWave: A Lightweight Hemisphere-Symmetric Waveform Network for Ensemble Classification of Harmful Brain Activity in EEG},
  author  = {Singh, Gagandeep and Dehzangi, Iman},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  note    = {under review},
  year    = {2026}
}
```

## License

To be added.

## Contact

Iman Dehzangi — i.dehzangi@rutgers.edu
