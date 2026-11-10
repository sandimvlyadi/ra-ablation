# Desain ablasi per RQ (usulan, belum disetujui, belum ada run)

**Dasar:** kandidat prioritas A/B di `02_kandidat_ablasi_fulltext.md`. **Tanggal:** 2026-11-10.
**Prinsip:** mengikuti pola E8/E9.
- Satu sumbu diubah per ablasi; komponen lain tetap.
- Aturan analisis di-commit sebelum run pertama.
- Pemilihan model hanya di validasi.
- Label uji tidak pernah diubah.
- Pembanding Holm dalam setiap RQ.

## 1. Konfigurasi rujukan dan unit biaya

- **Rujukan:** ResNet34 + ImageNet (A1) + balanced softmax (B2), resep augmentasi E8 `base`. Pembandingnya run E8 yang sudah ada dengan seed yang sama, jadi tidak perlu dilatih ulang.
- **Satu arm** = 100% label × 3 seed + 10% label × 3 draw (seed 42), persis seperti E8. Jumlahnya 6 run.
- **Perkiraan biaya per arm ResNet34:** sekitar 1,9 jam GPU. Dasarnya median E3 A1 B2: 30,9 menit pada 100% dan 5,9 menit pada 10%. Ini perkiraan kasar dan belum diukur ulang untuk arm baru.
- **RQ3:** setiap arm langsung dievaluasi pada RheumaVIT (transfer langsung). Waktunya hanya inferensi, beberapa menit. Few-shot (5 dan 20 pasien, seperti E4) hanya untuk arm terbaik per RQ.

## 2. Arm per RQ

Kolom **Kode**: *ada flag* = cukup memakai flag `train.py` yang sudah ada; *kode baru* = perlu diimplementasikan dan diuji dulu.

### RQ1: SSL dan pre-training (sumbu encoder; kepala tetap B2)

Sudah ada di tesis (dipakai ulang, tidak dilatih lagi): scratch, ImageNet, MAE, DINO, DINO + ChronoCon, LLRD.

| Arm | Isi | Kode | Biaya (perkiraan) |
|---|---|---|---|
| R1-1 | Pre-training antara supervised pada radiograf tangan RSNA bone age, lalu fine-tune (M005) | kode baru (regresi bone age), lalu `--init-from` | ±1 jam pre-train + 1,9 jam |
| R1-2 | SupCon + CE dengan negatif berbobot jarak grade (modifikasi dari M219/M082) | kode baru (loss) | ±2,5 jam |
| R1-3 | SSL normalitas: dilatih pada sendi grade 0, jarak ke pusat normal dipakai sebagai fitur tambahan (M197) | kode baru | ±2 jam pre-train + 1,9 jam |
| R1-4 | Strategi fine-tuning: freeze (linear probe), LP-FT, gradual unfreeze, pada A1 dan A4 | sebagian ada (`--freeze-backbone`, `--init-from`, `--layer-decay`) | 3 strategi × 2 encoder × 1,9 jam ≈ 11 jam |
| R1-5 | SKELEX (ViT-L): linear probe, lalu fine-tune + LLRD (M015) | kode baru (loader bobot) | belum bisa diperkirakan; **ukur 1 epoch dulu** (ViT-L di RTX 3060 kemungkinan lebih dari 5× ResNet34) |
| R1-6 (B) | Penyelarasan representasi progresi (MMD) sebagai alternatif ChronoCon (M335) | kode baru | ±3 jam |

### RQ2: imbalance dan ordinalitas (sumbu kepala/loss; encoder A1, diulang pada A4 untuk 2 arm terbaik)

Sudah ada di tesis: B0–B6, E7 undersampling, E8 augmentasi.

| Arm | Isi | Kode | Biaya |
|---|---|---|---|
| R2-1 | **Oversampling saja** (B1 tanpa class weight) | **ada flag** (`--oversample`) | 1,9 jam |
| R2-2 | **Class weight saja** (B1 tanpa oversampling) | **ada flag** (`--class-weight`) | 1,9 jam |
| R2-3 | Focal loss (γ = 2) | kode baru (kecil) | 1,9 jam |
| R2-4 | Focal + oversampling (temuan M041) | kode baru (kecil) + flag | 1,9 jam |
| R2-5 | Kepala cumulative link / cut-point (M089) | kode baru (kepala) | 1,9 jam |
| R2-6 | Soft label ordinal (beta atau Gaussian, M266/C047) | kode baru (kecil) | 1,9 jam |
| R2-7 (B) | B2 + kepala CORN tambahan sebagai regularizer (C015) | kode baru (kecil) | 1,9 jam |
| R2-P | **Pasca-latih, tanpa training:** ambang per kelas disetel di val, kNN di atas embedding, TTA | kode analisis | ±0 GPU |

### RQ3: ketahanan RAM-H1200 → RheumaVIT

Tidak ada arm sendiri. Untuk setiap arm di RQ1, RQ2, RQ4, dan RQ5 dilaporkan *domain drop* (BACC4/QWK internal − eksternal). Dua tambahan:
- **R3-1:** test-time BN adaptation pada RheumaVIT tanpa label (M095), diterapkan pada arm terbaik per RQ. Hampir tanpa biaya GPU.
- **R3-2:** few-shot 5 dan 20 pasien untuk arm terbaik per RQ. Biayanya seperti E4.

### RQ4: kalibrasi dan rujukan selektif

| Arm | Isi | Kode | Biaya |
|---|---|---|---|
| R4-P | **Pasca-latih:** deep ensemble antar seed, RAPS, conformal Mondrian per grade, vector/Dirichlet scaling | kode analisis | ±0 GPU |
| R4-1 | MC dropout di head + kurva risk-coverage (M130, M110) | kode baru (kecil) | 1,9 jam |
| R4-2 (B) | Evidential deep learning (M270) | kode baru | 1,9 jam |

### RQ5 (baru): label noise dan teacher-student

| Arm | Isi | Kode | Biaya |
|---|---|---|---|
| R5-0 | Prediksi out-of-fold pada train (5 fold, 100%, 1 seed) sebagai bahan CleanLab | kode baru (loop fold) | ±2,6 jam |
| R5-1 | CleanLab **relabel** di train (label uji tetap) | kode baru + library `cleanlab` | 1,9 jam |
| R5-2 | CleanLab **prune** (buang sampel yang diragukan) | idem | 1,9 jam |
| R5-3 | CleanLab **bobot turun** | idem | 1,9 jam |
| R5-4 | Semi-supervised mean teacher / consistency pada sendi tanpa label, rezim 10% dan 25% (M061, M185) | kode baru | ±4 jam |
| R5-5 | Pemisahan sampel berdasarkan reliabilitas: confidence + inkonsistensi antarkunjungan (M200; modifikasi dari temuan 2026-11-09) | kode baru | 1,9 jam |
| R5-6 | Label smoothing ε = 0,1 | kode baru (kecil) | 1,9 jam |
| R5-7 (B) | Self-distillation dari ensemble seed | kode baru | 1,9 jam |
| R5-E | **Evaluasi tambahan:** terhadap distribusi 3 pembaca RheumaVIT, di samping label konsensus | kode analisis | ±0 GPU |

## 3. Total perkiraan (tanpa SKELEX)

| Bagian | Arm berlatih | Jam GPU (kasar) |
|---|---|---|
| Pasca-latih (R2-P, R4-P, R3-1, R5-E) | 0 | ±0 |
| RQ2 prioritas A (R2-1 … R2-6) + 2 arm diulang pada A4 | 8 | ±15 |
| RQ5 prioritas A (R5-0 … R5-6) | 7 | ±16 |
| RQ1 prioritas A tanpa SKELEX (R1-1 … R1-4) | 9 | ±20 |
| RQ4 (R4-1) | 1 | ±2 |
| **Jumlah** | **25** | **±53 jam** (sekitar 6–7 malam GPU) |
| SKELEX (R1-5) | 2 | ukur dulu |
| Arm prioritas B | 4 | ±9 |

## 4. Urutan yang diusulkan

1. **Pasca-latih** (tanpa GPU, hasil cepat): ambang di val, kNN, TTA, deep ensemble, RAPS/Mondrian, vector scaling, evaluasi soft label pembaca.
2. **RQ2:** R2-1 dan R2-2 bisa langsung jalan karena flag-nya sudah ada. Setelah itu R2-3 sampai R2-6.
3. **RQ5:** R5-0, lalu R5-1 sampai R5-6. RQ ini terkait langsung dengan diagnosis label tidak stabil.
4. **RQ1:** R1-4 (sebagian ada flag), R1-1, R1-2, R1-3. Untuk SKELEX ukur 1 epoch dulu, baru diputuskan.
5. **RQ4:** R4-1.
6. **RQ3:** dilaporkan bersama setiap batch, few-shot untuk arm terbaik.

Setiap tahap dikerjakan dengan langkah yang sama: tulis kode dan test → commit aturan pra-spesifikasi → run → analisis → lapor.

## 5. Yang perlu diputuskan penulis (atau dosen)

1. Apakah susunan arm dan prioritas ini sudah sesuai harapan?
2. Apakah boleh mulai dari langkah 1 (pasca-latih) dan R2-1/R2-2, yang tidak butuh kode training baru?
3. Ukuran sampel: tetap 3 seed + 3 draw seperti E8, atau ditambah? Untuk efek kecil, memori E8 menunjukkan noise antar-ulangan sekitar setengah dari efek yang terdeteksi.
