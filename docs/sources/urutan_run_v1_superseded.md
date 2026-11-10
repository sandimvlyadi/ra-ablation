# Urutan run per RQ di folder baru (rencana, belum dieksekusi)

**Tanggal:** 2026-11-10.
**Dasar:** `02_kandidat_ablasi_fulltext.md` dan `03_desain_ablasi.md`.
**Perkiraan waktu:** kasar, dari waktu E3. 1 arm = 6 run (100% label × seed 42/1337/2026 + 10% label × draw 0/1/2, seed 42). Satu arm ResNet34 ≈ 1,9 jam. Satu arm ViT-S ≈ 2,9 jam (asumsi 1,5× ResNet34, belum diukur).

## Struktur folder baru (usulan)

```
<folder-baru>/                 ← siap dijadikan repo sendiri
  code/        salinan paket ramscore dari experiments @ b6e0135 (dicatat), lalu dikembangkan di sini
  artifacts/   encoder SSL lama yang diimpor, read-only + sha256: DINO R34, DINO ViT-S,
               DINO+ChronoCon R34/ViT-S, MAE ViT-S (hasil E1, tidak di-pre-train ulang)
  data/        symlink read-only ke RAM-H1200 dan RheumaVIT (data tidak disalin)
  rules/       aturan pra-spesifikasi per RQ, di-commit sebelum run pertama RQ itu
  runs/rq0 … rq5/   hasil run per RQ
  reports/     analisis per RQ + tabel ablasi gabungan
```

Run lama di `experiments/` tidak dipakai sebagai pembanding. Semua pembanding dilatih ulang di folder baru dengan resep yang sama dan augmentasi yang di-seed. Dengan begitu satu tabel ablasi hanya berisi run dari satu tempat.

Penamaan run: `rq<n>-<nomor>_<encoder>_<head>_f<fraksi>_d<draw>_s<seed>`.

## RQ0: rujukan bersama (jalan pertama)

| Kode | Isi | Run | Jam |
|---|---|---|---|
| RQ0-01 | ResNet34 + ImageNet + B2 (balanced softmax), augmentasi `base` dengan seed | 6 | 1,9 |

Run ini menjadi pembanding RQ1 (sumbu encoder) dan RQ2 (sumbu kepala/loss).

## RQ1: SSL dan pre-training (kepala tetap B2)

Yang ditanyakan: metode pre-training mana yang paling membantu, dan apakah modifikasinya menambah gain?

| Kode | Encoder / metode | Asal | Run | Jam |
|---|---|---|---|---|
| RQ1-01 | Scratch (ResNet34) | ulang | 6 | 1,9 |
| (RQ0-01) | ImageNet supervised (ResNet34) | rujukan | — | — |
| RQ1-02 | DINO (ResNet34) | artefak lama + fine-tune ulang | 6 | 1,9 |
| RQ1-03 | DINO + ChronoCon (ResNet34) | artefak lama + fine-tune ulang | 6 | 1,9 |
| RQ1-04 | ImageNet ViT-S + LLRD | ulang | 6 | 2,9 |
| RQ1-05 | MAE ViT-S + LLRD | artefak lama + fine-tune ulang | 6 | 2,9 |
| RQ1-06 | DINO ViT-S + LLRD | artefak lama + fine-tune ulang | 6 | 2,9 |
| RQ1-07 | **Baru:** pre-training antara pada radiograf tangan RSNA bone age, lalu fine-tune (M005) | perlu dataset RSNA (Kaggle, unduh dengan akun Anda) | 6 + pre-train | 2,9 |
| RQ1-08 | **Baru, modifikasi:** SupCon + CE dengan negatif berbobot jarak grade | kode baru | 6 | 2,5 |
| RQ1-09 | **Baru:** SSL normalitas pada sendi grade 0 (M197) | kode baru | 6 + pre-train | 3,9 |
| RQ1-10 | Strategi fine-tuning: linear probe dan LP-FT, masing-masing pada ImageNet dan encoder SSL terbaik dari RQ1-02 s.d. 06 | flag ada | 4 × 6 | 7,6 |
| RQ1-11 | **Baru:** SKELEX (ViT-L foundation model MSK), linear probe lalu fine-tune + LLRD (M015) | kode loader; **ukur 1 epoch dulu** | 2 × 6 | ? |
| RQ1-12 (B) | **Baru, modifikasi:** penyelarasan progresi antarkunjungan (MMD) sebagai alternatif ChronoCon (M335) | kode baru | 6 | 3 |
| RQ1-13 (B) | **Baru:** RAD-DINO / BiomedCLIP di-fine-tune (M157 hanya menguji versi beku) | kode loader | 2 × 6 | ±6 |

**Subtotal prioritas A tanpa SKELEX:** sekitar 31 jam.

## RQ2: class imbalance dan ordinalitas (encoder tetap ImageNet ResNet34)

Yang ditanyakan: cara penanganan imbalance dan kepala ordinal mana yang terbaik?

| Kode | Isi | Asal | Run | Jam |
|---|---|---|---|---|
| RQ2-01 | CE polos, tanpa penanganan imbalance (baseline sumbu) | flag ada | 6 | 1,9 |
| RQ2-02 | B1 asli: class weight + oversampling, kepala ordinal BCE | flag ada | 6 | 1,9 |
| RQ2-03 | **Oversampling saja** | flag ada | 6 | 1,9 |
| RQ2-04 | **Class weight saja** | flag ada | 6 | 1,9 |
| (RQ0-01) | B2 balanced softmax / logit adjustment | rujukan | — | — |
| RQ2-05 | B4 CORN + logit adjustment | flag ada | 6 | 1,9 |
| RQ2-06 | B6 decoupled cRT | flag ada | 6 | 1,9 |
| RQ2-07 | Undersampling grade 0 (E7 `rotate`) | flag ada | 6 | 1,9 |
| RQ2-08 | **Baru:** focal loss (γ = 2) | kode kecil | 6 | 1,9 |
| RQ2-09 | **Baru:** focal + oversampling (M041) | kode kecil | 6 | 1,9 |
| RQ2-10 | **Baru:** kepala cumulative link / cut-point (M089) | kode kecil | 6 | 1,9 |
| RQ2-11 | **Baru:** soft label ordinal, beta/Gaussian (M266, C047) | kode kecil | 6 | 1,9 |
| RQ2-12 (B) | **Baru, modifikasi:** B2 + kepala CORN sebagai regularizer (C015) | kode kecil | 6 | 1,9 |
| RQ2-13 | Interaksi: 2 arm terbaik RQ2 diulang pada encoder terbaik RQ1 | — | 2 × 6 | ±4–6 |
| RQ2-P | **Pasca-latih (tanpa GPU):** ambang per kelas disetel di validasi, kNN di atas embedding, TTA | kode analisis | 0 | 0 |

**Subtotal:** sekitar 25–27 jam.

## RQ3: ketahanan RAM-H1200 → RheumaVIT (untuk hasil RQ1 dan RQ2, sesuai arahan dosen)

| Kode | Isi | Run | Jam |
|---|---|---|---|
| RQ3-01 | Transfer langsung: setiap run RQ0, RQ1, dan RQ2 memprediksi RheumaVIT, lalu dihitung *domain drop* | inferensi | ±1 |
| RQ3-02 | Test-time BN adaptation tanpa label pada arm terbaik RQ1 dan RQ2 (M095) | inferensi | ±0,2 |
| RQ3-03 | Few-shot 5 dan 20 pasien untuk arm terbaik RQ1 dan RQ2, resep E4 (3 run sumber × 3 draw × 2 ukuran) | 36 run pendek | ±3 |
| RQ3-04 | Jangkar pembaca: model vs masing-masing dari 3 radiolog RheumaVIT | analisis | 0 |

## RQ4: kalibrasi dan rujukan selektif (di atas run RQ0 dan arm terbaik RQ2)

| Kode | Isi | Asal | Run | Jam |
|---|---|---|---|---|
| RQ4-P1 | Temperature scaling + log-prior (metode lama, dihitung ulang) | analisis | 0 | 0 |
| RQ4-P2 | Vector / Dirichlet scaling | analisis | 0 | 0 |
| RQ4-P3 | Conformal ordinal APS (metode lama), RAPS, Mondrian per grade | analisis | 0 | 0 |
| RQ4-P4 | Deep ensemble antar seed sebagai skor rujukan | analisis | 0 | 0 |
| RQ4-01 | **Baru:** MC dropout di head + kurva risk-coverage (M130, M110) | kode kecil | 6 | 1,9 |
| RQ4-02 (B) | **Baru:** evidential deep learning (M270) | kode baru | 6 | 1,9 |

Semua dilaporkan internal dan di RheumaVIT.

## RQ5 (baru): label noise, label correction, teacher-student (encoder dan kepala = rujukan RQ0)

| Kode | Isi | Asal | Run | Jam |
|---|---|---|---|---|
| RQ5-00 | Prediksi out-of-fold: 5 fold per pasien **di dalam train**, 100%, seed 42 | kode baru | 5 | 2,6 |
| RQ5-01 | CleanLab **relabel** label train (val/test tidak diubah) | kode baru | 6 | 1,9 |
| RQ5-02 | CleanLab **prune** | kode baru | 6 | 1,9 |
| RQ5-03 | CleanLab **bobot turun** | kode baru | 6 | 1,9 |
| RQ5-04 | Label smoothing ε = 0,1 | kode kecil | 6 | 1,9 |
| RQ5-05 | **Modifikasi:** pemisahan sampel berdasarkan reliabilitas (confidence + inkonsistensi antarkunjungan; M200) | kode baru | 6 | 1,9 |
| RQ5-06 | Semi-supervised mean teacher / consistency, rezim 10% dan 25% (pasien train di luar draw = tanpa label; M061, M185) | kode baru | 2 × 3 | ±4 |
| RQ5-07 (B) | Self-distillation dari ensemble seed | kode baru | 6 | 1,9 |
| RQ5-E | Evaluasi tambahan: RheumaVIT per pembaca dan terhadap distribusi pembaca | analisis | 0 | 0 |

**Subtotal:** sekitar 16 jam.

## Total

| Bagian | Jam GPU (kasar) |
|---|---|
| RQ0 | 1,9 |
| RQ1 (tanpa SKELEX) | ±31 |
| RQ2 | ±27 |
| RQ3 | ±4 |
| RQ4 | ±2–4 |
| RQ5 | ±16 |
| **Jumlah** | **±82 jam** (sekitar 10 malam GPU) + SKELEX |

**Opsi hemat:** sekitar 15 jam bisa dihemat dengan tidak melatih ulang arm yang sudah ada di `experiments/` (scratch, DINO, MAE, B1, B4, B6, E7) dan cukup mengimpor hasilnya. Konsekuensinya, run lama itu memakai augmentasi tanpa seed, jadi tidak sepenuhnya sebanding (lihat E8: pengulangan resep yang sama bergeser +0,001 sampai +0,024).
