# Machine Learning Project — Wine Quality & Imagenette

Repositori tugas Machine Learning yang menggabungkan tiga pendekatan dalam **satu proyek Python terpadu dengan satu `uv` environment**:

| Bagian | Metode Utama | Dataset | Tujuan |
| :--- | :--- | :--- | :--- |
| Supervised Learning | Linear Regression | UCI Wine Quality | Memprediksi skor `quality` dari karakteristik fisikokimia wine |
| Unsupervised Learning | Mean Shift Clustering | UCI Wine Quality | Mengelompokkan wine berdasarkan kemiripan fitur tanpa memasukkan `quality` |
| Deep Learning | DenseNet-169 & EfficientNet-B1 (Pretrained ImageNet) | Imagenette (Fullsize / 160px) | Klasifikasi gambar 10 kelas dengan arsitektur yang dipilih via parameter `--model` |

---

## Struktur Repositori

```text
.
├── Makefile
├── pyproject.toml
├── uv.lock
├── README.md
├── data/
│   ├── wine-quality/
│   │   ├── winequality-red.csv
│   │   ├── winequality-white.csv
│   │   └── winequality.names
│   ├── imagenette2/              # Letakkan dataset Imagenette fullsize di sini
│   └── imagenette2-160/          # Fallback jika memakai versi 160px
├── outputs/
│   ├── linear_regression/
│   ├── mean_shift/
│   └── deep_learning/
├── src/ml_project/
│   ├── cli.py                    # Entrypoint CLI terpadu (`ml-project`)
│   ├── core/                     # Shared IO, paths, seeding/device, plotting
│   ├── supervised/               # Pipeline Linear Regression
│   ├── unsupervised/             # Pipeline Mean Shift Clustering
│   └── deep_learning/            # DataLoader, trainer, evaluator, model registry
│       └── models/
│           ├── base.py
│           ├── densenet.py       # DenseNet-169 (default input 224x224)
│           ├── efficientnet.py   # EfficientNet-B1 (default input 240x240)
│           └── registry.py       # Factory & registry arsitektur
└── tests/                        # Suite pengujian pytest
```

---

## Prasyarat & Instalasi (`uv`)

- Python `3.11` – `3.13`
- `uv` terpasang

Sinkronisasi dependensi dari root repositori:

```bash
uv sync
# atau via Makefile:
make sync
```

---

## Dataset

1. **Wine Quality** (`data/wine-quality/`)
   - Sudah tersedia di repositori (`winequality-white.csv` dan `winequality-red.csv`, pemisah `;`).
2. **Imagenette Fullsize** (`data/imagenette2/`)
   - Ekstrak dataset Imagenette fullsize ke `data/imagenette2/` sehingga terdapat subfolder `train/` dan `val/`.
   - Jika `data/imagenette2/` belum ada, pipeline otomatis mencari `data/imagenette2-320/` lalu `data/imagenette2-160/`, atau gunakan `--data-dir <path>`.

---

## Menjalankan Eksperimen

### Via `Makefile`

```bash
make help          # Lihat daftar target
make sync          # Install dependensi dengan uv
make check         # Jalankan ruff lint + pytest
make regression    # Jalankan Linear Regression
make clustering    # Jalankan Mean Shift Clustering
make dl-train MODEL=densenet169 EPOCHS=10 BATCH_SIZE=32
make dl-train MODEL=efficientnet_b1 EPOCHS=10 BATCH_SIZE=32
make dl-eval CHECKPOINT=outputs/deep_learning/<run_id>/best_model.pth
```

### Via CLI Langsung (`uv run ml-project`)

#### 1. Linear Regression

```bash
uv run ml-project linear-regression
# atau dengan dataset red wine:
uv run ml-project linear-regression --data-path data/wine-quality/winequality-red.csv --no-show-plots
```

#### 2. Mean Shift Clustering

```bash
uv run ml-project mean-shift
# atau dengan bandwidth manual:
uv run ml-project mean-shift --data-path data/wine-quality/winequality-red.csv --bandwidth 1.2 --no-show-plots
```

#### 3. Deep Learning (`densenet169` & `efficientnet_b1`)

Ukuran input gambar otomatis menyesuaikan standar arsitektur kecuali di-override dengan `--image-size`:
- `densenet169`: resize `256` -> crop `224x224`, pretrained `DenseNet169_Weights.DEFAULT`
- `efficientnet_b1`: resize `256` -> crop `240x240`, pretrained `EfficientNet_B1_Weights.DEFAULT`

```bash
# DenseNet-169 (default, pretrained ImageNet)
uv run ml-project dl train \
  --model densenet169 \
  --weights imagenet \
  --data-dir data/imagenette2 \
  --epochs 10 \
  --batch-size 32 \
  --lr 0.001

# EfficientNet-B1 (pretrained ImageNet, fine-tune head saja)
uv run ml-project dl train \
  --model efficientnet_b1 \
  --weights imagenet \
  --freeze-backbone \
  --data-dir data/imagenette2 \
  --epochs 10 \
  --batch-size 32 \
  --lr 0.001

# Evaluasi checkpoint terbaik
uv run ml-project dl evaluate \
  --checkpoint outputs/deep_learning/<run_id>/best_model.pth \
  --data-dir data/imagenette2
```

---

## Testing & Code Quality

```bash
uv run ruff check src tests
uv run pytest
# atau sekaligus:
make check
```
