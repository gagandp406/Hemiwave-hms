# CoRaL — Committee-of-Latent-Raters Learning for HMS (design, Oct 19 2026)

Replaces PRISM-EEG as the main line of work. PRISM alone reached 0.2732 on the 6,350 ≥10-vote rows (VIPEEGNet 0.2217) and was only useful as a blend partner (blend 0.2177). The ask now: survey everything published on HMS and the IIIC literature, rate unexplored directions, pick one, and iterate on it with real runs.

## 1. What the field already knows

### 1.1 Where the data comes from (Jing et al. 2023, Neurology + supplement)
These facts drive the choice below:
- 124 raters scored 50,697 ten-second segments. Twenty fellowship-trained experts each labelled ≥1,000 segments.
- **High-quality set:** 9,129 segments with ≥10 labels, all from those 20 experts.
- **Lower-quality set:** the rest, from 119 raters, mostly fewer labels.
- **Stationary periods (SP):** experts labelled the central segment of a stationary period, and that label was copied to every other segment in the period. That is how HMS reaches 106,800 rows, and why 3rd place could "undo" the expansion back to ~6,350 originals.
- **Rater calibration:** the paper models each expert's calibration as a single per-class logit shift θ (over-calling > 0, under-calling < 0).
- **SPaRCNet's training:** hard majority labels on expert data, plus per-expert nearest-neighbour "label spreading" in a UMAP to simulate every expert labelling every segment.
- HMS vote counts are 1–7 or 10–28, never 8 or 9. The ≥10-vote rows are the expert pool; the test set behaves like them.

### 1.2 Top Kaggle solutions (private LB; 1st = 0.2723, VIPEEGNet single model = 0.2725)
| Place | Inputs / model | Label handling | Distinctive trick |
|---|---|---|---|
| 1 (Sony) | 18 bipolar EEG cropped at 3 lengths → 2D CNN; Morlet / Paul CWT → MaxViT 512²; superlets + learned 1D conv features → SwinV2/ConvNeXt etc. | stage 1 all rows (one member dropped 1-vote rows), stage 2 >9 votes; +0.02 per vote before normalising | entmax output (+0.004 LB); non-negative linear stacking |
| 2 | raw EEG reshaped as image → EfficientNet-B5; STFT → X3D (3D CNN) | two-stage | 3D CNN over channel position; L/R mirror |
| 3 | GPU mel-spectrogram 2D CNN; 1D conv + Squeezeformer on raw EEG | trained/validated on ~6.35k "original" HQ rows after undoing the SP expansion | clip(−1024,1024)/32 instead of per-sample norm; L/R channel ordering; NN blend with per-class bias |
| 4 | 4-member ensemble; bilzard's 1D EEG model treats left/right symmetrically (CV 0.2477, private 0.3277 alone) | not checked | channel-quality masks |
| 8 (Deotte) | one "mega" model | pseudo labels on <10-vote rows | — |
| 9 | contrastive feature handling | — | — |

What every top team shared:
- two-stage training (all rows, then ≥10 votes);
- left/right flip augmentation;
- patient GroupKFold;
- KL loss;
- ensembles of diverse inputs.

The label tricks (stage split, drop 1-vote rows, +0.02 offset, pseudo labels, vote weights, separate heads per vote tier) are all hand-tuned answers to one problem: low-vote rows come from a different rater pool than the test labels.

### 1.3 Research literature that matters
- **IIIC / SPaRCNet** (Jing 2023): DenseNet on 16-channel bipolar 10 s EEG; per-expert pseudo labels.
- **ProtoPMed-EEG** (Barnett et al., NEJM AI 2024): interpretable "this looks like that" prototypes.
- **VIPEEGNet** (npj Digital Medicine 2025): EfficientNetV2-B3 on an EEG-image; our baseline.
- **EEG foundation models** (BIOT, LaBraM, CBraMod, EEGPT): masked pretraining on TUH. Independent benchmarks (EEG-FM-Bench, Brain4FMs) show mixed results and no consistent win over task-specific supervised models on clinical tasks. None reports HMS KLD competitive with the Kaggle top.
- **Learning from crowds:**
  - Dawid–Skene; Raykar 2010; crowd layer (Rodrigues & Pereira 2018).
  - Guan, Gulshan, Dai & Hinton 2018, "Who said what": one head per known doctor plus learned mixing weights, which improved diabetic-retinopathy classification.
  - Soft-label aggregation (Uma 2020; PLOS ONE 2025).
- **Distribution outputs:** mixture of softmaxes (Yang 2018); entmax (Peters 2019); Dirichlet-multinomial overdispersion.
- **Retrieval-augmented prediction:** kNN-LM (Khandelwal 2020). SPaRCNet's UMAP label spreading is a training-time cousin.

## 2. Where our remaining error is (from `model-comparison.md`)
- LRDA recall is 22–38% for every model and GRDA 42–52%. These are the classes with the most expert disagreement.
- All models are slightly *less* confident than the experts (mean max-probability 0.67 vs mean top vote share 0.71).
- Blending a weaker but different model (PRISM) still gains 0.004. VIPEEGNet's errors aren't irreducible, but architecture alone (PRISM) did not reach them.

## 3. Candidate directions, rated
Scores are 1–5; higher is better (for cost, 5 = cheapest).

| # | Direction | Novelty for HMS | Expected gain | Cost | Risk (5 = low) | Verdict |
|---|---|---|---|---|---|---|
| 1 | **CoRaL**: model the labels as a latent-rater committee with two rater pools and a vote-count likelihood; plus a kNN vote-memory | 4 | 3 | 5 (first tests need no training) | 3 | **Chosen** |
| 2 | Privileged long-context teacher (train EEGs are minutes long; test is 50 s + 10-min spectrogram), distilled into a 50 s student | 4 | 2–3 | 2 | 2 | Keep for later |
| 3 | In-domain self-supervised pretraining on unlabelled HMS EEG time | 2–3 | 2 | 2 | 3 | Was PRISM's next step |
| 4 | Fine-tune an EEG foundation model (CBraMod / LaBraM) | 2 | 1–2 | 3 | 2 | Benchmarks don't favour it |
| 5 | Stationary-period multi-view consistency loss | 3 | 1–2 | 4 | 3 | Random sub-window sampling already gets most of it |
| 6 | 1st-place-style CWT + MaxViT second model | 1 | 3 (in a blend) | 3 | 4 | Not new; useful only as a blend partner |
| 7 | Generative (diffusion) augmentation of rare LRDA | 2 | 1 | 1 | 2 | No |
| 8 | ACNS-factorised morphology head (PRISM) | — | negative | — | — | Already tested: −0.010 vs a flat head |

**Why 1 wins:**
- It targets the documented data-generating process: two rater pools, per-expert logit-shift calibration, SP label copying.
- The top teams handled that process with hand-tuned heuristics worth about 0.01 in their ablations.
- It is model-agnostic, so it can improve VIPEEGNet itself rather than only blend with it.
- It can be screened in minutes on archived outputs before any GPU training.

## 4. The method

### 4.1 Committee output layer
Shared encoder → perception logits z(x). The head then builds:
- **R latent raters** (R = 8 by default). Rater r votes with pᵣ(x) = softmax(τᵣ · (z(x) + b_pool + θᵣ)), where θᵣ is a per-class logit shift (Jing's calibration model) and τᵣ a temperature.
- **Two pools:** low-vote (119 raters) and expert (20 experts). Each pool has its own mixing weights πₚₒₒₗ over the R latent raters (optionally sample-specific, as in Guan et al. but with latent raters).
- **Predicted vote share** μₚₒₒₗ(x) = Σᵣ πₚₒₒₗ,ᵣ(x) · pᵣ(x). At test time it outputs μ_expert.

A mixture of shifted softmaxes can represent split votes (e.g. 50% LPD / 50% SZ) that a single softmax only reaches by flattening everything. This is the same motivation as entmax's +0.004, from the opposite side.

### 4.2 Likelihood on raw votes
- **Dirichlet-multinomial** on the vote counts, with a learned per-pool concentration κ: −log DirMult(v | κ·μₚₒₒₗ).
- Rows with more votes count more, automatically. Overdispersion absorbs the stationary-period label copying.
- The low-vote rows train the shared encoder and the low-pool mixture, without leaking their rater biases into the expert prediction.
- One principled objective replaces: two-stage training, dropping 1-vote rows, +0.02 offsets, vote weights and the earlier EBVD target denoising.

### 4.3 kNN vote memory (optional module)
- For a query, the k nearest labelled training rows (cosine distance in the encoder's feature space) contribute their raw vote counts, weighted by similarity. Neighbours act as extra raters.
- The result is blended with μ_expert.
- k, temperature, key set (all rows vs expert rows), blend weight and arithmetic vs geometric mixing are all chosen leave-one-fold-out.

### 4.4 Honest novelty statement
- The building blocks are published: crowd modelling, mixtures of softmaxes, DirMult, kNN-LM.
- Not found in HMS solutions or the IIIC papers:
  - latent-rater committees with pool-specific mixing trained on vote counts;
  - DirMult with a per-pool learned concentration as the replacement for multi-stage label heuristics;
  - inference-time vote-count retrieval.
- It is incremental, not a new paradigm. Recheck prior art before publishing.

## 5. Experimental plan (each step must beat the previous best on identical rows)
**Decision rule:**
- pooled ≥10-vote KLD with a paired patient-bootstrap CI, against both VIPEEGNet and the same-protocol control;
- differences < 0.003 need the CI (fold spread ±0.007);
- every hyperparameter is chosen on training rows (inner patient split) or leave-one-fold-out.

**Track A — archived OOF logits (no GPU, ~5–15 min).** `run_coral.sh logits`
- Heads take VIPEEGNet's 6 logits as input, fitted on the other four folds' OOF rows (all 20,183 rows, both pools).
- Control: A0, affine recalibration on ≥10-vote rows (Phase 0 style).
- Ladder: A two-stage KL → B multinomial on all rows → C + pool bias → D + 8 latent raters → E + DirMult → F + gating → G + HQ refinement.
- Tests the output layer and likelihood as a structured recalibration. Expected −0.001 to −0.004 vs VIPEEGNet.

**Track B — frozen VIPEEGNet features (≈20 min extraction + ~30 min heads).** `run_coral.sh extract`, then `features`
- 1536-d penultimate features per fold model. Heads start at VIPEEGNet's own final layer and are anchored to it.
- Same ladder, plus kNN vote memory on top of every rung.
- Caveat: training-row features come from the model that saw them, so compare rungs with each other more than with VIPEEGNet.

**Phase 2 — end-to-end CoRaL on the VIPEEGNet backbone (~6 h / variant).** Only if B shows D/E/G beating A with a CI excluding 0.
- Replace VIPEEGNet's head and two-stage KL with the committee head and DirMult, one stage on all rows.
- Everything else unchanged, so the comparison is clean.
- Keras implementation written after Track B results.

**Phase 3 — scale.**
- kNN memory on end-to-end features.
- CoRaL head on a second backbone (PRISM-wave or a CWT model) for a stronger blend.
- Kaggle late submission for public/private LB.

**Kill criteria:** if Track B's best CoRaL rung isn't at least 0.002 better than control A (CI excluding 0) and retrieval adds nothing, drop CoRaL and move to direction 2 (privileged long-context teacher).

## 6. Success criteria
- **Main:** an end-to-end CoRaL VIPEEGNet below 0.2217 pooled on the 6,350 rows, with a patient-bootstrap CI excluding 0. Then confirm on Kaggle private LB (VIPEEGNet 0.2725, 1st place 0.2723).
- **Secondary:** CoRaL-VIPEEGNet + CoRaL-second-model blend below the current best blend, 0.2177.

## 7. Files
- Code: `~/hms/coral/`:
  - `coral_lib.py`: head, losses, training, kNN;
  - `coral_eval.py`: votes recovery, KLD, bootstrap, archive-schema saving;
  - `screen_logits.py`, `extract_vip_features.py`, `adapter_template.py`, `screen_features.py`, `run_coral.sh`.
- Outputs:
  - `~/hms/fair_comparison/coral_screen/<tag>/oof/fold{k}_oof[_all].npz` (VIPEEGNet schema, so `load_oof` / `compare_models.ipynb` work);
  - `screen_*_results.csv/json`.

## 8. Learning map (what to know to work on this well)
1. **Clinical EEG:** 10–20 system; bipolar "double banana" vs average montages; ACNS 2021 critical-care terminology (PD vs RDA, L vs G, the ictal–interictal–injury continuum, the 10 s / evolution criteria for seizure). Source: Hirsch et al. 2021 J Clin Neurophysiol; learningeeg.com.
2. **Signal processing:** band-pass/notch filtering and zero-phase filtfilt; STFT vs CWT (Morlet, Paul) vs superlets (Moca et al. 2021); time–frequency resolution trade-offs; why 0.5–40 Hz.
3. **HMS data provenance:** stationary-period expansion; two rater pools; why ≥10-vote CV tracks the LB; patient-level GroupKFold; leakage through the copied SP labels.
4. **Learning from distributions:** KL vs cross-entropy (identical gradients for fixed targets); label distribution learning; soft labels; why the expected-KL-optimal prediction is the committee mean.
5. **Learning from crowds:** Dawid–Skene; EM with annotator confusion matrices; crowd layer; Guan et al. 2018; calibration as a logit shift.
6. **Probability models for counts:** multinomial vs Dirichlet-multinomial; overdispersion; empirical Bayes; conjugate posteriors.
7. **Output layers:** softmax bottleneck and mixture of softmaxes; sparsemax/entmax; temperature scaling and calibration (ECE, Brier).
8. **Retrieval-augmented prediction:** kNN-LM; similarity-weighted label smoothing; leakage control for retrieval (patient-disjoint keys).
9. **Evaluation:** paired patient bootstrap; nested / leave-one-fold-out tuning; why fold-mean and pooled KLD differ; non-negative stacking.
10. **Architectures in play:** EfficientNetV2, MaxViT/SwinV2 on scalograms, 1D CNN + GRU/Squeezeformer, EEG foundation models (LaBraM, CBraMod) and their benchmark evidence.
