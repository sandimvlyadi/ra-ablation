# RQ1: pre-training dan SSL vs inisialisasi ImageNet

*Post hoc*: test RAM-H1200 sudah dipakai di repo lama, sehingga hasil ini tidak dipakai untuk memilih konfigurasi tesis; semua pilihan memakai validasi. Aturan dikunci sebelum run pertama: `rules/rq1_prespecification.md`. Holm per keluarga (F1–F4) dengan ukuran tetap; kontras yang belum dijalankan masuk Holm dengan p = 1. p-value QWK dan ICC tidak dikoreksi. Putusan: bermakna = Holm p < 0,05 dan |selisih| ≥ 0,05; kecil = Holm p < 0,05 dan |selisih| < 0,05; selain itu tidak terbukti (bukan bukti kesetaraan).

ImageNet ViT-S memakai `augreg_in1k` (ImageNet-1k), bukan in21k seperti E2/E6, jadi angkanya tidak sebanding langsung dengan E2/E6. Encoder SSL E1 melihat citra val tanpa label.

## Pilihan berbasis validasi (rata-rata BACC4 validasi)

- **Metode terbaik yang sudah ada, 100%** (sementara): Scratch ResNet34. Urutan: Scratch ResNet34 0.554, DINO ResNet34 0.501, DINO ViT-S (LLRD) 0.500, MAE ViT-S (LLRD) 0.499, ImageNet ViT-S (LLRD) 0.481, ImageNet ResNet34 (rujukan) 0.476, DINO + ChronoCon ResNet34 0.473. Belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14, rq1-18b.
- **Metode terbaik yang sudah ada, 10%** (sementara): Scratch ResNet34. Urutan: Scratch ResNet34 0.423, ImageNet ResNet34 (rujukan) 0.416, DINO ResNet34 0.401, ImageNet ViT-S (LLRD) 0.391, MAE ViT-S (LLRD) 0.380, DINO + ChronoCon ResNet34 0.379, DINO ViT-S (LLRD) 0.343. Belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14, rq1-18b.
- **Metode terbaik selain SKELEX, 100%** (sementara): Scratch ResNet34. Urutan: Scratch ResNet34 0.554, DINO ResNet34 0.501, DINO ViT-S (LLRD) 0.500, MAE ViT-S (LLRD) 0.499, ImageNet ViT-S (LLRD) 0.481, ImageNet ResNet34 (rujukan) 0.476, DINO + ChronoCon ResNet34 0.473. Belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14.
- **Metode terbaik selain SKELEX, 10%** (sementara): Scratch ResNet34. Urutan: Scratch ResNet34 0.423, ImageNet ResNet34 (rujukan) 0.416, DINO ResNet34 0.401, ImageNet ViT-S (LLRD) 0.391, MAE ViT-S (LLRD) 0.380, DINO + ChronoCon ResNet34 0.379, DINO ViT-S (LLRD) 0.343. Belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14.
- **SSL terbaik (6 run, kedua fraksi)** (sementara): DINO ResNet34. Urutan: DINO ResNet34 0.451, MAE ViT-S (LLRD) 0.440, DINO + ChronoCon ResNet34 0.426, DINO ViT-S (LLRD) 0.422. Belum selesai: rq1-08, rq1-10.

## F1: pre-training / SSL vs rujukan ImageNet ResNet34 (Holm atas 16 kontras)

| Label | Arm | Pembanding | Run | BACC4 pembanding | BACC4 arm | Selisih [95% CI] | p mentah | Holm p | Putusan | Selisih QWK | p | Selisih ICC | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100% | Scratch ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.475 | 0.500 | +0.025 [-0.010, +0.062] | 0.147 | 1.000 | tidak terbukti | -0.088 | 0.001 | -0.070 | 0.011 |
| 10% | Scratch ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.369 | 0.328 | -0.041 [-0.080, +0.002] | 0.063 | 1.000 | tidak terbukti | -0.218 | 0.000 | -0.453 | 0.000 |
| 100% | DINO ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.475 | 0.445 | -0.030 [-0.075, +0.014] | 0.193 | 1.000 | tidak terbukti | -0.075 | 0.001 | -0.060 | 0.078 |
| 10% | DINO ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.369 | 0.376 | +0.007 [-0.057, +0.065] | 0.848 | 1.000 | tidak terbukti | +0.028 | 0.415 | -0.071 | 0.421 |
| 100% | ImageNet → bone age RSNA ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | ImageNet → bone age RSNA ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | SimCLR ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | SimCLR ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | BYOL ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | BYOL ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | SupCon + CE ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | SupCon + CE ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | SSL normalitas grade 0 ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | SSL normalitas grade 0 ResNet34 | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | SKELEX ViT-L fine-tune (LLRD) | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | SKELEX ViT-L fine-tune (LLRD) | ImageNet ResNet34 (rujukan) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |

## F2: antar-metode dan arsitektur (Holm atas 14 kontras)

| Label | Arm | Pembanding | Run | BACC4 pembanding | BACC4 arm | Selisih [95% CI] | p mentah | Holm p | Putusan | Selisih QWK | p | Selisih ICC | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100% | DINO + ChronoCon ResNet34 | DINO ResNet34 | 3 | 0.445 | 0.438 | -0.007 [-0.033, +0.019] | 0.610 | 1.000 | tidak terbukti | -0.022 | 0.318 | +0.007 | 0.788 |
| 10% | DINO + ChronoCon ResNet34 | DINO ResNet34 | 3 | 0.376 | 0.341 | -0.035 [-0.079, +0.009] | 0.127 | 1.000 | tidak terbukti | -0.057 | 0.072 | +0.056 | 0.482 |
| 100% | ImageNet ViT-S (LLRD) (tercampur resep: ViT-S memakai LLRD + warmup) | ImageNet ResNet34 (rujukan) | 3 | 0.475 | 0.468 | -0.006 [-0.047, +0.032] | 0.779 | 1.000 | tidak terbukti | -0.010 | 0.698 | -0.041 | 0.240 |
| 10% | ImageNet ViT-S (LLRD) (tercampur resep: ViT-S memakai LLRD + warmup) | ImageNet ResNet34 (rujukan) | 3 | 0.369 | 0.364 | -0.006 [-0.063, +0.053] | 0.803 | 1.000 | tidak terbukti | +0.053 | 0.022 | -0.032 | 0.559 |
| 100% | MAE ViT-S (LLRD) | ImageNet ViT-S (LLRD) | 3 | 0.468 | 0.473 | +0.004 [-0.037, +0.051] | 0.825 | 1.000 | tidak terbukti | -0.061 | 0.102 | -0.005 | 0.809 |
| 10% | MAE ViT-S (LLRD) | ImageNet ViT-S (LLRD) | 3 | 0.364 | 0.389 | +0.025 [-0.047, +0.104] | 0.535 | 1.000 | tidak terbukti | +0.022 | 0.473 | +0.070 | 0.455 |
| 100% | DINO ViT-S (LLRD) | ImageNet ViT-S (LLRD) | 3 | 0.468 | 0.492 | +0.024 [-0.014, +0.067] | 0.202 | 1.000 | tidak terbukti | -0.066 | 0.034 | -0.038 | 0.546 |
| 10% | DINO ViT-S (LLRD) | ImageNet ViT-S (LLRD) | 3 | 0.364 | 0.367 | +0.003 [-0.057, +0.064] | 0.990 | 1.000 | tidak terbukti | -0.016 | 0.608 | +0.019 | 0.723 |
| 100% | SimCLR ResNet34 | DINO ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | SimCLR ResNet34 | DINO ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | BYOL ResNet34 | DINO ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | BYOL ResNet34 | DINO ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | SKELEX ViT-L fine-tune (LLRD) | metode terbaik selain SKELEX | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14) | — | — | — | — |
| 10% | SKELEX ViT-L fine-tune (LLRD) | metode terbaik selain SKELEX | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14) | — | — | — | — |

## F3: modifikasi (vs versi asli dan vs metode terbaik yang sudah ada) (Holm atas 8 kontras)

| Label | Arm | Pembanding | Run | BACC4 pembanding | BACC4 arm | Selisih [95% CI] | p mentah | Holm p | Putusan | Selisih QWK | p | Selisih ICC | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100% | SupCon berbobot jarak grade ResNet34 (modifikasi) | SupCon + CE ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | SupCon berbobot jarak grade ResNet34 (modifikasi) | SupCon + CE ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | SupCon berbobot jarak grade ResNet34 (modifikasi) | metode terbaik yang sudah ada | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14, rq1-18b) | — | — | — | — |
| 10% | SupCon berbobot jarak grade ResNet34 (modifikasi) | metode terbaik yang sudah ada | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14, rq1-18b) | — | — | — | — |
| 100% | Gate normalitas 0 vs ≥1 ResNet34 (modifikasi) | SSL normalitas grade 0 ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | Gate normalitas 0 vs ≥1 ResNet34 (modifikasi) | SSL normalitas grade 0 ResNet34 | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 100% | Gate normalitas 0 vs ≥1 ResNet34 (modifikasi) | metode terbaik yang sudah ada | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14, rq1-18b) | — | — | — | — |
| 10% | Gate normalitas 0 vs ≥1 ResNet34 (modifikasi) | metode terbaik yang sudah ada | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-07, rq1-08, rq1-10, rq1-12, rq1-14, rq1-18b) | — | — | — | — |

## F4: strategi fine-tuning (Holm atas 10 kontras)

| Label | Arm | Pembanding | Run | BACC4 pembanding | BACC4 arm | Selisih [95% CI] | p mentah | Holm p | Putusan | Selisih QWK | p | Selisih ICC | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100% | Linear probe ImageNet ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.475 | 0.474 | -0.000 [-0.035, +0.038] | 0.935 | 1.000 | tidak terbukti | -0.144 | 0.000 | -0.139 | 0.001 |
| 10% | Linear probe ImageNet ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.369 | 0.383 | +0.014 [-0.030, +0.054] | 0.477 | 1.000 | tidak terbukti | +0.033 | 0.423 | +0.021 | 0.649 |
| 100% | LP-FT ImageNet ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.475 | 0.464 | -0.011 [-0.045, +0.027] | 0.469 | 1.000 | tidak terbukti | -0.037 | 0.042 | -0.007 | 0.673 |
| 10% | LP-FT ImageNet ResNet34 | ImageNet ResNet34 (rujukan) | 3 | 0.369 | 0.344 | -0.026 [-0.051, +0.008] | 0.114 | 1.000 | tidak terbukti | -0.005 | 0.851 | +0.007 | 0.913 |
| 100% | Linear probe SSL terbaik | SSL terbaik | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-08, rq1-10) | — | — | — | — |
| 10% | Linear probe SSL terbaik | SSL terbaik | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-08, rq1-10) | — | — | — | — |
| 100% | LP-FT SSL terbaik | SSL terbaik | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-08, rq1-10) | — | — | — | — |
| 10% | LP-FT SSL terbaik | SSL terbaik | — | — | — | — | — | — | tidak dijalankan (pilihan validasi belum final; kandidat belum selesai: rq1-08, rq1-10) | — | — | — | — |
| 100% | SKELEX ViT-L linear probe | SKELEX ViT-L fine-tune (LLRD) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |
| 10% | SKELEX ViT-L linear probe | SKELEX ViT-L fine-tune (LLRD) | — | — | — | — | — | — | tidak dijalankan (arm belum dijalankan) | — | — | — | — |

## Validasi, test, dan epoch terpilih (rata-rata run)

Kolom test − val adalah cek deskriptif optimisme test, bukan uji. Validasi hanya 36 pasien.

| Label | Arm | Run | Val BACC4 | Val QWK | Test BACC4 | Test − val | Epoch terpilih |
|---|---|---|---|---|---|---|---|
| 100% | ImageNet ResNet34 (rujukan) | 3 | 0.476 | 0.415 | 0.475 | -0.001 | 28.0 |
| 100% | Scratch ResNet34 | 3 | 0.554 | 0.353 | 0.500 | -0.055 | 41.7 |
| 100% | DINO ResNet34 | 3 | 0.501 | 0.424 | 0.445 | -0.056 | 20.3 |
| 100% | DINO + ChronoCon ResNet34 | 3 | 0.473 | 0.357 | 0.438 | -0.035 | 24.3 |
| 100% | ImageNet ViT-S (LLRD) | 3 | 0.481 | 0.390 | 0.468 | -0.012 | 16.7 |
| 100% | MAE ViT-S (LLRD) | 3 | 0.499 | 0.389 | 0.473 | -0.026 | 8.3 |
| 100% | DINO ViT-S (LLRD) | 3 | 0.500 | 0.348 | 0.492 | -0.008 | 5.7 |
| 100% | Linear probe ImageNet ResNet34 | 3 | 0.492 | 0.287 | 0.474 | -0.018 | 8.7 |
| 100% | LP-FT ImageNet ResNet34 | 3 | 0.492 | 0.424 | 0.464 | -0.029 | 18.7 |
| 10% | ImageNet ResNet34 (rujukan) | 3 | 0.416 | 0.238 | 0.369 | -0.047 | 33.3 |
| 10% | Scratch ResNet34 | 3 | 0.423 | 0.045 | 0.328 | -0.095 | 29.3 |
| 10% | DINO ResNet34 | 3 | 0.401 | 0.264 | 0.376 | -0.025 | 28.3 |
| 10% | DINO + ChronoCon ResNet34 | 3 | 0.379 | 0.199 | 0.341 | -0.038 | 34.3 |
| 10% | ImageNet ViT-S (LLRD) | 3 | 0.391 | 0.260 | 0.364 | -0.027 | 15.7 |
| 10% | MAE ViT-S (LLRD) | 3 | 0.380 | 0.237 | 0.389 | +0.009 | 24.0 |
| 10% | DINO ViT-S (LLRD) | 3 | 0.343 | 0.177 | 0.367 | +0.024 | 31.0 |
| 10% | Linear probe ImageNet ResNet34 | 3 | 0.387 | 0.219 | 0.383 | -0.003 | 15.7 |
| 10% | LP-FT ImageNet ResNet34 | 3 | 0.369 | 0.186 | 0.344 | -0.025 | 33.0 |

## Recall per grade (test)

| Label | Arm | Run | Proporsi prediksi grade 0 | Recall 0 | Recall 1 | Recall 2 | Recall 3 | Recall 5 |
|---|---|---|---|---|---|---|---|---|
| 100% | ImageNet ResNet34 (rujukan) | 3 | 0.766 | 0.862 | 0.348 | 0.127 | 0.187 | 0.546 |
| 100% | Scratch ResNet34 | 3 | 0.691 | 0.774 | 0.352 | 0.176 | 0.493 | 0.541 |
| 100% | DINO ResNet34 | 3 | 0.696 | 0.779 | 0.402 | 0.096 | 0.147 | 0.470 |
| 100% | DINO + ChronoCon ResNet34 | 3 | 0.736 | 0.804 | 0.311 | 0.118 | 0.133 | 0.459 |
| 100% | ImageNet ViT-S (LLRD) | 3 | 0.780 | 0.870 | 0.318 | 0.130 | 0.173 | 0.514 |
| 100% | MAE ViT-S (LLRD) | 3 | 0.756 | 0.836 | 0.317 | 0.129 | 0.120 | 0.645 |
| 100% | DINO ViT-S (LLRD) | 3 | 0.642 | 0.730 | 0.481 | 0.090 | 0.213 | 0.705 |
| 100% | Linear probe ImageNet ResNet34 | 3 | 0.648 | 0.735 | 0.372 | 0.208 | 0.187 | 0.536 |
| 100% | LP-FT ImageNet ResNet34 | 3 | 0.743 | 0.834 | 0.359 | 0.146 | 0.147 | 0.514 |
| 10% | ImageNet ResNet34 (rujukan) | 3 | 0.765 | 0.819 | 0.229 | 0.054 | 0.240 | 0.137 |
| 10% | Scratch ResNet34 | 3 | 0.434 | 0.456 | 0.389 | 0.076 | 0.187 | 0.208 |
| 10% | DINO ResNet34 | 3 | 0.865 | 0.902 | 0.146 | 0.010 | 0.253 | 0.251 |
| 10% | DINO + ChronoCon ResNet34 | 3 | 0.753 | 0.790 | 0.219 | 0.051 | 0.240 | 0.071 |
| 10% | ImageNet ViT-S (LLRD) | 3 | 0.877 | 0.915 | 0.079 | 0.089 | 0.200 | 0.246 |
| 10% | MAE ViT-S (LLRD) | 3 | 0.778 | 0.829 | 0.280 | 0.033 | 0.107 | 0.432 |
| 10% | DINO ViT-S (LLRD) | 3 | 0.835 | 0.876 | 0.200 | 0.039 | 0.040 | 0.372 |
| 10% | Linear probe ImageNet ResNet34 | 3 | 0.600 | 0.682 | 0.468 | 0.109 | 0.093 | 0.279 |
| 10% | LP-FT ImageNet ResNet34 | 3 | 0.733 | 0.784 | 0.287 | 0.095 | 0.013 | 0.246 |
