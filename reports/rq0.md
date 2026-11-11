# RQ0: rujukan bersama dan cek arsitektur (ResNet34 vs ConvNeXt-Tiny)

*Post hoc*: test RAM-H1200 sudah dipakai di repo lama, sehingga hasil ini tidak dipakai untuk memilih konfigurasi tesis. Aturan dikunci sebelum run pertama: `rules/rq0_prespecification.md`. Holm dihitung atas 2 kontras primer BACC4; p-value lain tidak dikoreksi.

## Kontras primer: ConvNeXt-Tiny − ResNet34 (rujukan)

| Label | Run | BACC4 ResNet34 | BACC4 ConvNeXt-Tiny | Selisih [95% CI] | Holm p | Putusan | Selisih QWK | p | Selisih ICC | p |
|---|---|---|---|---|---|---|---|---|---|---|
| 100% | 3 | 0.475 | 0.495 | +0.020 [-0.012, +0.048] | 0.392 | tidak terbukti | +0.052 | 0.011 | +0.016 | 0.475 |
| 10% | 3 | 0.369 | 0.387 | +0.018 [-0.023, +0.053] | 0.392 | tidak terbukti | +0.102 | 0.001 | +0.007 | 0.935 |

## Uji reproduksi: rujukan di repo ini vs run E8 `base` dengan draw/seed yang sama (tanpa putusan)

| Label | Run | BACC4 lama | BACC4 baru | Selisih [95% CI] | p | Selisih QWK | p |
|---|---|---|---|---|---|---|---|
| 100% | 3 | 0.475 | 0.475 | +0.000 [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| 10% | 3 | 0.369 | 0.369 | +0.000 [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |

Per run (|selisih BACC4| ≥ 0.02 = RQ1 ditahan sampai penyebabnya dicari):

| Label | Desain | Prediksi identik | Selisih BACC4 | Tanda |
|---|---|---|---|---|
| 100% | d0_s1337 | 1.000 | +0.000 | ok |
| 100% | d0_s2026 | 1.000 | +0.000 | ok |
| 100% | d0_s42 | 1.000 | +0.000 | ok |
| 10% | d0_s42 | 1.000 | +0.000 | ok |
| 10% | d1_s42 | 1.000 | +0.000 | ok |
| 10% | d2_s42 | 1.000 | +0.000 | ok |

## Validasi dan epoch terpilih (rata-rata run)

| Label | Arm | Run | Val BACC4 | Val QWK | Epoch terpilih |
|---|---|---|---|---|---|
| 100% | ResNet34 (rujukan) | 3 | 0.476 | 0.415 | 28.0 |
| 100% | ConvNeXt-Tiny | 3 | 0.507 | 0.512 | 15.7 |
| 10% | ResNet34 (rujukan) | 3 | 0.416 | 0.238 | 33.3 |
| 10% | ConvNeXt-Tiny | 3 | 0.370 | 0.273 | 13.3 |

## Recall per grade (test)

| Label | Arm | Run | Proporsi prediksi grade 0 | Recall 0 | Recall 1 | Recall 2 | Recall 3 | Recall 5 |
|---|---|---|---|---|---|---|---|---|
| 100% | ResNet34 (rujukan) | 3 | 0.766 | 0.862 | 0.348 | 0.127 | 0.187 | 0.546 |
| 100% | ConvNeXt-Tiny | 3 | 0.781 | 0.889 | 0.376 | 0.153 | 0.120 | 0.601 |
| 10% | ResNet34 (rujukan) | 3 | 0.765 | 0.819 | 0.229 | 0.054 | 0.240 | 0.137 |
| 10% | ConvNeXt-Tiny | 3 | 0.844 | 0.899 | 0.221 | 0.033 | 0.227 | 0.240 |
