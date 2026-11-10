# Konfigurasi teknis bersama (disepakati 2026-11-10/11)

## Data
- **Latih / val / test:** ROI sendi RAM-H1200, split resmi per pasien (`ramscore.data`, root `train.py: DEFAULT_ROOT`). Model memprediksi 5 grade erosi asli. Metrik dihitung pada 4 level 0/1/2/≥3 (BACC4 primer).
- **Rezim label:**
  - 100% label × seed 42, 1337, 2026
  - 10% label per pasien × draw (`--fraction-seed`) 0, 1, 2 dengan seed 42

  Jadi **1 arm = 6 run**. Draw dan seed-nya sama dengan E3/E8 di repo lama.
- **Eksternal:** RheumaVIT, 13 sendi yang dipetakan 1:1 (`ramscore.rheumavit`). Grade 4 dan 5 dipetakan ke level 3. Referensinya median bawah dari 3 pembaca. Manifest ada di `data/derived/rheumavit/manifest.csv`.

## Resep rujukan (RQ0-01 = arm `base` E8)
```
uv run python -m ramscore.train --name <nama> --out ../runs/rq0 --amp --n-boot 0 \
  --model resnet34 --pretrained --head softmax --logit-adjust 1 \
  --label-fraction {1.0|0.1} --fraction-seed {0|1|2} --seed {42|1337|2026} \
  --epochs 50 --min-steps 2000 --aug base --seed-aug
```
- ResNet34 dengan bobot ImageNet (timm), input 224, batch 32.
- AdamW, lr 1e-4, weight decay 1e-3, cosine.
- Checkpoint dipilih dengan `--monitor qwk` (default) pada validasi.
- Augmentasi `base`: ShiftScaleRotate (rotate 10, p 0,5) + RandomBrightnessContrast (p 0,5), di-seed per worker. Val/test tidak diaugmentasi.

## Aturan perubahan per arm
- Hanya satu faktor yang berubah terhadap rujukan.
- Flag baru diberi default mati, supaya run lama dan rujukan tidak berubah. Setiap flag baru diberi unit test, lalu smoke test dengan `--limit` sebelum masuk antrean.

## Evaluasi
- **Internal:** BACC4 (primer), QWK, recall per grade, ICC skor total tangan, proporsi prediksi grade 0.
- **Eksternal:** transfer langsung (tanpa pelatihan ulang), metrik yang sama, ditambah drop = internal − eksternal. Few-shot 5/20 pasien dengan resep E4 (lr 1e-5, 500 step) hanya untuk arm terbaik.

## Statistik
- **Kontras:** arm vs pembanding pada 100% dan 10%. Holm di dalam satu RQ.
- **Uji:** bootstrap pasien berpasangan, 10.000 replikasi, run ikut diambil ulang (`ramscore.compare`).
- **Putusan BACC4:**
  - "bermakna": Holm p < 0,05 dan |selisih| ≥ 0,05
  - "kecil": Holm p < 0,05 dan |selisih| < 0,05
  - "tidak terbukti": selain itu
- **Modifikasi** dibandingkan dengan versi aslinya dan dengan metode terbaik yang ada.
- Semua hasil *post hoc* terhadap test RAM-H1200 (sudah dipakai di repo lama).
