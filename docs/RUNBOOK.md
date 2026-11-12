# RUNBOOK: perintah → kode yang dipanggil → output

**Konvensi:**
- Semua perintah dijalankan dari `code/` (di sana ada `pyproject.toml`) dengan `uv run`. Jalur output relatif ke `code/`, jadi `../runs/…` = `ra-ablation/runs/…`.
- Status:
  - ✅ = sudah ada dan teruji
  - 🛠 = akan ditulis (baris ini diperbarui saat kode selesai, beserta nama fungsinya)
- Setiap baris baru ditambahkan **saat** perintah atau kode baru dibuat (aturan 8 di `CLAUDE.md`).

## 0. Lingkungan dan integritas
| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| Membuat lingkungan | `uv sync --group dev --group analysis` | `pyproject.toml`, `uv.lock` | `code/.venv/` | ✅ 2026-11-11 |
| Test | `uv run pytest -q` | `code/tests/test_*.py` | 230 passed | ✅ 2026-11-11 |
| Cek artefak encoder | `cd ../artifacts && sha256sum -c SHA256SUMS` | — | OK per file | ✅ |
| Cek ROI RheumaVIT | `cd ../data/derived && sha256sum -c ../../artifacts/RHEUMAVIT_ROI_SHA256SUMS` (salinan di git) | — | OK per file | ✅ |
| (Hanya bila ROI hilang) buat ulang ROI RheumaVIT | `uv run python -m ramscore.rheumavit --src <folder AIRI asli> --out ../data/derived/rheumavit` | `rheumavit.py: main()` → ROI 13 sendi + `manifest.csv` | `data/derived/rheumavit/` | ✅ (kode lama) |

## 1. Satu run training (dipakai semua arm)
**Perintah** (contoh RQ0-01, 100% label, seed 42):
```
uv run python -m ramscore.train --name rq0-01_resnet34_b2_f1.0_d0_s42 --out ../runs/rq0 --amp --n-boot 0 \
  --model resnet34 --pretrained --head softmax --logit-adjust 1 --label-fraction 1.0 --fraction-seed 0 --seed 42 \
  --epochs 50 --min-steps 2000 --aug base --seed-aug
```

**Rantai kode** (`code/src/ramscore/`):
- `train.py: main()`
  - parse argumen
  - `data.py`: baca ROI + label dari `DEFAULT_ROOT`, split resmi; `--label-fraction`/`--fraction-seed` memilih pasien; `AUGMENTATIONS[--aug]`; sampler (`--oversample`, `--undersample-zero`)
  - `models.py`: backbone timm (`--model`, `--pretrained` atau `--init-from`), kepala (`--head`), `Criterion` (`--class-weight`, `--logit-adjust`)
  - loop epoch: validasi setiap epoch, checkpoint terbaik menurut `--monitor` (qwk)
  - prediksi test, lalu `metrics.py: compute_metrics()`

**Output** `runs/rq0/<nama>/`:
- `args.json`: semua argumen
- `history.csv`: loss + metrik val per epoch
- `model_best.pt`: checkpoint terbaik di val
- `test_predictions.csv`: gt/pred per sendi
- `metrics.json`: best_epoch, metrik val dan test
- `undersample.json`: hanya bila `--undersample-zero`

**Flag per arm:**
- sudah ada: `--model`, `--pretrained`, `--init-from`, `--freeze-backbone`, `--layer-decay`, `--warmup`, `--head`, `--class-weight`, `--oversample`, `--logit-adjust`, `--undersample-zero`, `--aug`, `--pair-weight`
- flag baru: 🛠 dicatat di bagian 6 saat dibuat

## 2. Antrean per tahap
| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| Lihat rencana + sisa jam | `bash ../scripts/run_rq<n>.sh` | skrip antrean (pola `run_e8.sh` di repo lama): daftar arm × 6 desain, melewati run yang sudah punya `metrics.json` | tabel arm / selesai / jam tersisa | 🛠 per tahap |
| Jalankan / lanjutkan setelah PC mati | `setsid nohup bash ../scripts/run_rq<n>.sh all >> ../runs/rq<n>.log 2>&1 < /dev/null &` | idem, memanggil `ramscore.train` per run | `runs/rq<n>/<run>/…`, `runs/rq<n>/logs/<run>.log`, `runs/rq<n>/queue.pid` | 🛠 |
| Hentikan | `kill -TERM -- -$(cat ../runs/rq<n>/queue.pid)` | — | antrean berhenti; run setengah jadi diulang saat dilanjutkan | 🛠 |

**RQ0** (aturan: `rules/rq0_prespecification.md`). Perintah dari root repo; skrip pindah sendiri ke `code/`.

| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| Status + sisa jam | `bash scripts/run_rq0.sh` | `scripts/run_rq0.sh: queue()` (2 arm × 6 desain, urutan arm berselang) | tabel arm / run selesai / jam tersisa (kasar) | ✅ 2026-11-11 |
| Smoke test | `OUT=<scratch> CONVNEXT=convnext_tiny.in12k_ft_in1k bash scripts/run_rq0.sh all --limit 640 --epochs 1 --min-steps 0 --workers 4` | idem → `ramscore.train` | 12/12 run dengan `metrics.json`, 2,5 menit | ✅ 2026-11-11 |
| Ukur 1 epoch (100% label) | `uv run python -m ramscore.train --name ep1_<m> --out <scratch> --amp --n-boot 0 --model <m> --pretrained --head softmax --logit-adjust 1 --label-fraction 1.0 --seed 42 --epochs 1 --min-steps 0 --aug base --seed-aug` | `train.py: main()` | ResNet34 54 s (puncak 2,7 GB VRAM); ConvNeXt-Tiny 91 s (4,3 GB) → ±1,9× ResNet34; perkiraan kasar RQ0 ±5,5 jam | ✅ 2026-11-11 |
| Jalankan / lanjutkan | `setsid nohup bash scripts/run_rq0.sh all >> runs/rq0.log 2>&1 < /dev/null &` | idem; `CONVNEXT=` memilih bobot RQ0-02 (default `convnext_tiny.fb_in1k`, sudah diunduh 2026-11-11) | `runs/rq0/rq0-0{1,2}_*/`, `runs/rq0/logs/`, `runs/rq0/queue.pid` | menunggu izin peluncuran |

**RQ1 tahap 1** (aturan: `rules/rq1_prespecification.md`, commit 4fccaab). Arm RQ1-01…06, 36 run.

| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| Status + sisa jam | `bash scripts/run_rq1.sh` | `scripts/run_rq1.sh: queue()`; flag per arm di `FLAGS` (ViT-S: `--layer-decay 0.85 --lr 1e-4 --warmup 0.1`; RQ1-04 `vit_small_patch16_224.augreg_in1k --pretrained`) | tabel arm / run selesai / jam tersisa (±13 jam kasar) | ✅ 2026-11-11 |
| Smoke test | `OUT=<scratch> bash scripts/run_rq1.sh all --limit 640 --epochs 1 --min-steps 0 --workers 4` | idem → `ramscore.train`; `train.py: load_backbone()` | 36/36 `metrics.json`, 9,5 menit (berbagi GPU dengan RQ0); log: encoder E1 termuat, hanya `fc.*` / `head.*` yang baru | ✅ 2026-11-11 |
| Jalankan / lanjutkan | `setsid nohup bash scripts/run_rq1.sh all >> runs/rq1.log 2>&1 < /dev/null &` | idem; cek `sha256sum -c artifacts/SHA256SUMS` dulu | `runs/rq1/rq1-0*_*/`, `runs/rq1/logs/`, `runs/rq1/queue.pid` | menunggu RQ0 selesai + izin |
| RQ1-17 tuning lr probe (adendum 1) | `setsid nohup bash scripts/run_rq1.sh tune17 >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: tune17()` → `train.py --pretrained --freeze-backbone --lr {1e-4,1e-3,1e-2}`, 10% d0 s42 | `runs/rq1/rq1-17tune_lr*_*/`; pilih lr dengan BACC4 **validasi** tertinggi, lalu catat di adendum + STATUS | smoke ✅ 2026-11-11; menunggu izin |
| RQ1-17a/b (ImageNet LP, LP-FT) | `LP_LR=<lr> setsid nohup bash scripts/run_rq1.sh lp17 >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: lp17_one()`: 17a `--freeze-backbone --lr $LP_LR`, lalu 17b `--init-from <17a>/model_best.pt --keep-head` (lr 1e-4) | `runs/rq1/rq1-17{a,b}_*/` | smoke ✅ 2026-11-11 (51/51 bersama tahap 1, args tahap 1 identik dengan sebelum refactor) |

## 3. Pre-training SSL (RQ1)
| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| DINO / MAE / ChronoCon (sudah ada sebagai artefak, **tidak** diulang) | `uv run python -m ramscore.pretrain --name <n> --method {dino,mae} --model {resnet34,vit_small_patch16_224} [--chrono-weight 1] --amp` | `pretrain.py: main()` → `pool.py` (kumpulan ROI tanpa label) → `ssl.py: DINO, MAE, chronocon_loss` | `runs/<n>/encoder.pt`, `checkpoint.pt`, `history.csv`, `encoder_ep*.pt` | ✅ (artefak di `artifacts/encoders/`) |
| Fine-tune dari encoder | `ramscore.train … --init-from ../artifacts/encoders/<e>/encoder.pt [--layer-decay 0.85 --warmup …] [--freeze-backbone]` | `train.py` → `models.py` memuat state dict encoder, kepala direset | sama dengan bagian 1 | ✅ |
| SimCLR / MoCo v3 / BYOL / iBOT / SupCon / normalitas / MMD | 🛠 `--method simclr\|mocov3\|byol\|ibot`, … | 🛠 `ssl.py` + `pretrain.py` | 🛠 | 🛠 |
| Bone age RSNA (RQ1-07) | 🛠 unduh: `kaggle …` (dengan izin), lalu pre-train regresi | 🛠 `rq1_boneage.py` | 🛠 | 🛠 |

## 4. Evaluasi eksternal dan pasca-latih
| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| Transfer langsung ke RheumaVIT untuk run mana pun | 🛠 `uv run python -m ramscore.rq3 direct --runs '../runs/rq<n>/*' --manifest ../data/derived/rheumavit/manifest.csv --out ../runs/rq3` | 🛠 `rq3.py` memakai `e4.py: direct()` / `to_levels()` / `analyse()` (repo lama, terikat ke konfigurasi K1–K4) | 🛠 `runs/rq3/direct/<run>/predictions.csv` | 🛠 |
| Few-shot 5/20 pasien | 🛠 `ramscore.rq3 fewshot …` | 🛠 memakai `e4.py: fewshot_patients()`, `finetune()` | 🛠 | 🛠 |
| Logit + kalibrasi + conformal | 🛠 `ramscore.rq4 …` | 🛠 memakai `e5.py: collect()`, `fit_temperature()`, `expansion_mass()`, `conformal_lambda()`, `evaluate()` | 🛠 `runs/rq4/…` | 🛠 |

## 5. Analisis statistik
| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| Kontras arm vs pembanding (bootstrap pasien berpasangan, run diambil ulang, Holm) | 🛠 `uv run --group analysis python -m ramscore.rq<n>` (pola `e8.py: analyse()`) | `compare.py: load_runs()`, `bootstrap()`, `compare()`, `holm()`, `to_markdown()` | 🛠 `runs/rq<n>_analysis/report.md` + `.json`, disalin ke `reports/` | 🛠 |
| RQ0: kontras primer + uji reproduksi vs E8 | `uv run --group analysis python -m ramscore.rq0 [--e8 <runs/e8 lama>]` | `rq0.py: analyse()` (memakai `e6abl.paired/contrast/add_holm`, `e7.prediction_stats`), `reproduction()`, `verdict()`, `report()` | `runs/rq0_analysis/report.md` + `rq0.json` (salin ke `reports/` setelah antrean selesai) | ✅ kode + 10 test (2026-11-11); menunggu run |
| RQ1: keluarga Holm F1–F4 (ukuran tetap), pilihan berbasis validasi, test − val | `uv run --group analysis python -m ramscore.rq1` | `rq1.py: selections()`, `select()`, `resolve()`, `family()`, `analyse()`, `report()` (memakai `e6abl.paired/contrast/add_holm`, `e7.prediction_stats`) | `runs/rq1_analysis/report.md` + `rq1.json`, disalin ke `reports/rq1.md` | ✅ 2026-11-12 (RQ1-01…06, 17a/b) |
| RQ1: celah train − val checkpoint terpilih (GPU, inferensi saja) | `uv run --group analysis python -m ramscore.rq1 gap` | `rq1.py: gap()` → `e8.py: gap(runs=…)` | `runs/rq1_analysis/gap.json` (dibaca otomatis oleh laporan) | 🛠 belum dijalankan |
| Kontras ad hoc dari CLI | `uv run python -m ramscore.compare <baseline> <arm…> --pattern '../runs/rq<n>/{}_*_f1.0_*' --out <file>` | `compare.py: main()` | `<file>.json`, `<file>.md` | ✅ (kode lama) |

## 6. Flag dan modul baru (diisi saat dibuat)
| Tanggal | Flag / modul | File : fungsi | Dipakai oleh arm | Test |
|---|---|---|---|---|
| 2026-11-11 | modul `rq0.py` | `rq0.py: analyse(), reproduction(), run_summary(), verdict(), report()` | RQ0-01, RQ0-02 | `tests/test_rq0.py` (10) |
| 2026-11-11 | `--freeze-backbone` kini boleh dengan `--pretrained` (linear probe ImageNet); flag baru `--keep-head` (dengan `--init-from`: muat checkpoint lengkap termasuk kepala, tahap 2 LP-FT) | `train.py: check_init_args()`, `main()`; `models.py: load_full()` | RQ1-17a–d | `tests/test_rq1.py` (11); smoke LP → LP-FT: backbone LP = ImageNet persis, LP-FT mengubah backbone |
| 2026-11-12 | modul `rq1.py`; `e8.gap()` menerima `runs=` (default tidak berubah) | `rq1.py: select(), selections(), resolve(), family(), analyse(), gap(), report()`; `e8.py: gap()` | RQ0-01, RQ1-* | `tests/test_rq1_analysis.py` (7) |
