# MAE-AST — Masked Autoencoding for Audio Spectrogram Transformers

Reimplementazione modulare in **PyTorch** di **MAE-AST** per il pretraining self-supervised di rappresentazioni audio.

Il progetto studia:

- **profondità dell'encoder, strategia di masking e mask ratio**;
- contributo delle loss **ricostruttiva** e **discriminativa/contrastiva**;
- **efficienza computazionale** ottenuta facendo elaborare all'encoder solo i token visibili.

Il pretraining finale è stato eseguito su un subset AudioSet di circa 500k clip; le rappresentazioni sono state valutate su **ESC-50** con protocollo ufficiale **5-fold**.

**Risultato migliore:** 12L + chunk 50% → **85.90 ± 1.49%** su ESC-50.  
**Massimo vantaggio misurato:** **2.13× speed-up** e **49.10%** di riduzione della peak GPU memory nel benchmark 12L/75%.

---

## Risultati

### Main ablation

| Run | Encoder | Masking | Ratio | ESC-50 accuracy |
|---|---:|---|---:|---:|
| M1 | 6L | random | 50% | 82.50 ± 3.05% |
| M2 | 6L | chunk | 50% | 84.45 ± 2.77% |
| M3 | 6L | random | 75% | 83.55 ± 2.25% |
| M4 | 6L | chunk | 75% | 84.35 ± 1.46% |
| M5 | 12L | random | 50% | 84.00 ± 2.88% |
| M6 | 12L | chunk | 50% | **85.90 ± 1.49%** |
| M7 | 12L | random | 75% | 85.15 ± 3.00% |
| M8 | 12L | chunk | 75% | 84.50 ± 1.94% |

I valori sono **media ± deviazione standard della popolazione** sui cinque fold ESC-50.

### Loss ablation

Configurazione controllata: **6L + chunk + 75%**.

| Objective | ESC-50 accuracy |
|---|---:|
| Reconstruction only | 80.60 ± 1.80% |
| Classification only | **84.60 ± 0.87%** |
| Joint | 84.35 ± 1.46% |

La loss congiunta è:

\[
\mathcal{L}=10\,\mathcal{L}_{rec}+\mathcal{L}_{cls}
\]

`L_cls` è una loss contrastiva **intra-clip** sulle patch mascherate: le 527 label AudioSet non vengono usate come target supervisionati nel pretraining.

### Efficiency benchmark

MAE-AST è confrontato con `FullSequenceProxy`: una baseline controllata che mantiene decoder e head, ma fa processare all'encoder la sequenza completa. **Non è una replica completa di SSAST.**

Protocollo: batch 32, 20 step di warm-up, 100 step misurati, 5 repliche, batch sintetici già residenti in memoria.

| Encoder | Mask ratio | Speed-up | Peak memory reduction |
|---|---:|---:|---:|
| 6L | 50% | 1.50× | 26.63% |
| 6L | 75% | 1.74× | 40.65% |
| 12L | 50% | 1.67× | 32.51% |
| 12L | 75% | **2.13×** | **49.10%** |

Il benchmark misura `forward + loss + backward + optimizer step` ed esclude disco, decoding audio, preprocessing e DataLoader.

---

## Implementazione

```text
WAV
→ mono / 16 kHz
→ log-Mel, 128 bin, frame 25 ms, hop 10 ms
→ 1024 × 128 nel pretraining
→ patch non sovrapposte 16 × 16
→ 512 token
→ masking
→ encoder sui soli token visibili
→ decoder sulla sequenza completa
```

Con 512 token:

- 50% masking → **256 token** nell'encoder;
- 75% masking → **128 token** nell'encoder.

Configurazione comune: embedding 768, 12 head, encoder 6/12 layer, decoder 2 layer, positional embedding sinusoidale 1D fisso, AdamW e mixed precision.

Nel fine-tuning il decoder viene scartato: `encoder → mean pooling → linear head → 50 classi ESC-50`.

---

## Repository

```text
.
├── configs/experiments/       # configurazioni portabili finali
├── datafiles/                 # stats AudioSet + manifest ESC-50
├── results/                   # artifact delle run pubblicate
├── scripts/                   # training, 5-fold runner, benchmark, campagne
├── src/mae_ast/               # implementazione
├── tools/                     # preparazione dataset e statistiche
└── pyproject.toml
```

File principali:

```text
configs/experiments/pretrain_final_6l.yaml
configs/experiments/pretrain_final_12l.yaml
configs/experiments/esc50_finetune_final_6l.yaml
configs/experiments/esc50_finetune_final_12l.yaml
scripts/run_final_pretrain_campaign_16ep.sh
scripts/run_final_esc50_campaign_16ep.sh
scripts/run_final_loss_ablation_16ep.sh
scripts/run_final_loss_ablation_esc50.sh
scripts/run_benchmark_final.sh
```

I `config.yaml` conservati sotto `results/` sono snapshot delle run originali e possono contenere path storici della macchina usata. Per nuove esecuzioni usare `configs/experiments/`.

---

# Riproducibilità

## 1. Setup

Requisiti dichiarati dal package: **Python >= 3.10**, PyTorch, TorchAudio, OmegaConf, NumPy e SoundFile.

```bash
git clone https://github.com/alessiopgg/deep_learning.git
cd deep_learning

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Per una GPU CUDA installare una build PyTorch/TorchAudio compatibile con il proprio sistema. L'ambiente usato per i risultati finali era:

```text
GPU      NVIDIA GeForce RTX 5090 (~32 GB)
Python   3.12.14
PyTorch  2.13.0+cu132
CUDA     13.2
```

Smoke test, senza dataset:

```bash
python scripts/smoke_test.py
```

I runner completi `run_final_*.sh` sono Bash: per la riproduzione integrale è consigliato Linux o un cluster Linux.

---

## 2. AudioSet 500k

Mirror usato: **`confit/audioset-16khz-wds`**, configurazione `500k`.  
Dataset: https://huggingface.co/datasets/confit/audioset-16khz-wds

Per il download installare la CLI Hugging Face:

```bash
python -m pip install -U huggingface_hub
```

<details>
<summary><strong>Preparazione completa di AudioSet</strong></summary>

### Download

```bash
export AUDIOSET_ROOT=/path/to/audioset500k
export HF_AUDIOSET_ROOT="$AUDIOSET_ROOT/hf"

mkdir -p "$AUDIOSET_ROOT/audio"
mkdir -p "$AUDIOSET_ROOT/test_audio"
mkdir -p "$AUDIOSET_ROOT/manifests"

hf download confit/audioset-16khz-wds \
  --repo-type dataset \
  --include "500k/train/*.tar" \
  --local-dir "$HF_AUDIOSET_ROOT"

hf download confit/audioset-16khz-wds \
  --repo-type dataset \
  --include "500k/test/*.tar" \
  --local-dir "$HF_AUDIOSET_ROOT"
```

Gli shard vengono salvati in:

```text
$HF_AUDIOSET_ROOT/500k/train/*.tar
$HF_AUDIOSET_ROOT/500k/test/*.tar
```

### Estrazione WAV e manifest

Train:

```bash
python tools/prepare_audioset.py \
  --tar-dir "$HF_AUDIOSET_ROOT/500k/train" \
  --audio-dir "$AUDIOSET_ROOT/audio" \
  --dataset-root "$AUDIOSET_ROOT" \
  --output "$AUDIOSET_ROOT/manifests/audioset_500k_train_full.json" \
  --extract-missing
```

Test ufficiale, estratto in una directory separata per evitare collisioni tra nomi di shard train/test:

```bash
python tools/prepare_audioset.py \
  --tar-dir "$HF_AUDIOSET_ROOT/500k/test" \
  --audio-dir "$AUDIOSET_ROOT/test_audio" \
  --dataset-root "$AUDIOSET_ROOT" \
  --output "$AUDIOSET_ROOT/manifests/audioset_500k_test_v2.json" \
  --extract-missing
```

### Split train/validation

Lo split è deterministico: SHA-256 di `seed:source_id`, seed 42, validation 2.5%.

```bash
python tools/split_audioset.py \
  --input-manifest "$AUDIOSET_ROOT/manifests/audioset_500k_train_full.json" \
  --train-output "$AUDIOSET_ROOT/manifests/audioset_500k_pretrain_train.json" \
  --val-output "$AUDIOSET_ROOT/manifests/audioset_500k_pretrain_val.json" \
  --test-manifest "$AUDIOSET_ROOT/manifests/audioset_500k_test_v2.json" \
  --val-ratio 0.025 \
  --seed 42 \
  --summary-output "$AUDIOSET_ROOT/manifests/audioset_500k_split_summary.json"
```

Conteggi della campagna finale:

```text
train ufficiale disponibile  490,548
pretraining train             478,284
validation interna             12,264
test ufficiale disponibile     16,950
```

### Statistiche di normalizzazione

Il file usato dagli esperimenti è già versionato:

```text
datafiles/audioset500k_train_stats_padded.json
```

```text
mean = -4.254134522085795
std  =  4.433377142672776
stats_source = padded_input
```

Per rigenerarlo a scopo di audit:

```bash
python tools/compute_stats.py \
  --manifest "$AUDIOSET_ROOT/manifests/audioset_500k_pretrain_train.json" \
  --dataset-root "$AUDIOSET_ROOT" \
  --output "$AUDIOSET_ROOT/manifests/audioset500k_train_stats_padded_recomputed.json" \
  --config configs/experiments/pretrain_final_6l.yaml \
  --mode pretrain \
  --stats-source padded_input
```

</details>

**Controllo importante.** La card Hugging Face riporta conteggi nominali che non coincidono necessariamente con gli shard effettivamente disponibili nel mirror. La campagna pubblicata usa i conteggi sopra: per una replica strettamente identica del corpus, verificare che i manifest generati coincidano. Se i numeri cambiano, si riproducono pipeline e protocollo ma non lo stesso snapshot dei dati.

La struttura minima attesa dai runner è:

```text
$AUDIOSET_ROOT/
├── audio/
└── manifests/
    ├── audioset_500k_pretrain_train.json
    └── audioset_500k_pretrain_val.json
```

---

## 3. ESC-50

Dataset ufficiale: https://github.com/karolpiczak/ESC-50

```bash
export ESC50_ROOT=/path/to/ESC-50
git clone https://github.com/karolpiczak/ESC-50.git "$ESC50_ROOT"
```

I manifest dei cinque fold ufficiali sono già inclusi in `datafiles/` e usano path relativi `audio/<file>.wav`. Ogni fold contiene 1600 clip di training e 400 di valutazione.

Rigenerazione opzionale dei manifest:

```bash
python tools/prepare_esc50.py \
  --dataset-root "$ESC50_ROOT" \
  --output-dir datafiles
```

---

## 4. Campagna principale M1–M8

```bash
CUDA_VISIBLE_DEVICES=0 \
AUDIOSET_ROOT="$AUDIOSET_ROOT" \
bash scripts/run_final_pretrain_campaign_16ep.sh
```

Il runner esegue la matrice `6L/12L × random/chunk × 50%/75%`.

Budget per modello:

```text
batch size       32
epochs           16
optimizer steps  239,136
warm-up          14,348
optimizer        AdamW
lr               1e-4
weight decay     0.01
```

Output di default:

```text
outputs/final_pretrain_16ep/
```

Il runner salta le run già complete e riprende automaticamente da `last.pt` quando disponibile.

---

## 5. ESC-50 5-fold di M1–M8

Dopo il pretraining:

```bash
CUDA_VISIBLE_DEVICES=0 \
ESC50_ROOT="$ESC50_ROOT" \
bash scripts/run_final_esc50_campaign_16ep.sh
```

Output di default:

```text
outputs/final_esc50_16ep/
```

Se il pretraining è stato scritto altrove:

```bash
CUDA_VISIBLE_DEVICES=0 \
ESC50_ROOT="$ESC50_ROOT" \
PRE_ROOT=/path/to/final_pretrain_16ep \
FT_ROOT=/path/to/final_esc50_16ep \
bash scripts/run_final_esc50_campaign_16ep.sh
```

---

## 6. Loss ablation

Pretraining:

```bash
CUDA_VISIBLE_DEVICES=0 \
AUDIOSET_ROOT="$AUDIOSET_ROOT" \
bash scripts/run_final_loss_ablation_16ep.sh
```

Fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 \
ESC50_ROOT="$ESC50_ROOT" \
bash scripts/run_final_loss_ablation_esc50.sh
```

Il confronto finale è `L1 reconstruction-only`, `L2 classification-only`, `M4 joint`.

---

## 7. Benchmark compute/memory

Non richiede dataset:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_benchmark_final.sh
```

Default del runner:

```text
BATCH_SIZE=32
WARMUP=20
STEPS=100
REPS=5
```

Output:

```text
outputs/benchmarks_final/
```

Parametri modificabili tramite variabili d'ambiente, ad esempio:

```bash
BATCH_SIZE=16 WARMUP=10 STEPS=50 REPS=3 \
bash scripts/run_benchmark_final.sh
```

---

## Provenance dei risultati

I checkpoint sono esclusi dal repository per dimensione. Gli artifact numerici usati per le tabelle sono invece versionati in:

```text
results/
├── main_ablation/
│   ├── pretrain/
│   └── esc50/
├── loss_ablation/
│   ├── pretrain/
│   └── esc50/
└── efficiency/
```

Sono conservati config effettive, log JSONL, risultati dei singoli fold, aggregati JSON/CSV e repliche individuali del benchmark.

### Limiti della riproducibilità

- i **checkpoint** originali non sono distribuiti: per rifare il downstream da zero occorre prima ripetere il pretraining;
- il repository non contiene un **lock file completo** di Python/PyTorch/CUDA: l'ambiente usato è riportato sopra, ma non è garantita una replica bit-per-bit su stack differenti;
- la riproduzione esatta di AudioSet richiede gli **stessi shard effettivamente disponibili** nel mirror usato dalla campagna finale.

---

## Riferimenti

- **MAE-AST** — Baade, Peng, Harwath, Interspeech 2022  
  https://arxiv.org/abs/2203.16691  
  https://github.com/AlanBaade/MAE-AST-Public

- **SSAST** — Gong, Lai, Chung, Glass  
  https://arxiv.org/abs/2110.09784  
  https://github.com/YuanGongND/ssast

- **AST** — Gong, Chung, Glass  
  https://arxiv.org/abs/2104.01778  
  https://github.com/YuanGongND/ast

- **Masked Autoencoders Are Scalable Vision Learners** — He et al.  
  https://arxiv.org/abs/2111.06377

- **AudioSet**  
  https://research.google.com/audioset/  
  Mirror usato: https://huggingface.co/datasets/confit/audioset-16khz-wds

- **ESC-50** — Piczak, ACM Multimedia 2015  
  https://github.com/karolpiczak/ESC-50

---

> Questa è una reimplementazione sperimentale di MAE-AST, non una replica 1:1 dell'implementazione fairseq originale. I confronti con il paper vanno interpretati nel setting adottato dal progetto: **AudioSet ~500k → ESC-50**.
