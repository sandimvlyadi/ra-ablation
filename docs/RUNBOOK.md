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
| RQ1-17c/d (LP, LP-FT dari SSL terbaik) | `LP_LR=1e-2 SSL_ARM=rq1-02 setsid nohup bash scripts/run_rq1.sh lp17ssl >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: lp_pair()` (dipakai juga oleh `lp17_one`), `lp17ssl()`: 17c `--init-from <encoder SSL> --freeze-backbone --lr $LP_LR`, lalu 17d `--init-from <17c>/model_best.pt --keep-head`; SSL terbaik dari `rq1.select(SSL_CANDIDATES)` | `runs/rq1/rq1-17{c,d}_*/` | smoke ✅ 2026-11-13 (12/12; argumen 17a/b sama dengan run lama) |

## 3. Pre-training SSL (RQ1)
| Langkah | Perintah | Kode | Output | Status |
|---|---|---|---|---|
| DINO / MAE / ChronoCon (sudah ada sebagai artefak, **tidak** diulang) | `uv run python -m ramscore.pretrain --name <n> --method {dino,mae} --model {resnet34,vit_small_patch16_224} [--chrono-weight 1] --amp` | `pretrain.py: main()` → `pool.py` (kumpulan ROI tanpa label) → `ssl.py: DINO, MAE, chronocon_loss` | `runs/<n>/encoder.pt`, `checkpoint.pt`, `history.csv`, `encoder_ep*.pt` | ✅ (artefak di `artifacts/encoders/`) |
| Fine-tune dari encoder | `ramscore.train … --init-from ../artifacts/encoders/<e>/encoder.pt [--layer-decay 0.85 --warmup …] [--freeze-backbone]` | `train.py` → `models.py` memuat state dict encoder, kepala direset | sama dengan bagian 1 | ✅ |
| RQ1-12 tuning λ SupCon (adendum 3) | `setsid nohup bash scripts/run_rq1.sh tune12 >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: tune12()` → `train.py --pretrained --supcon-weight {0.1,0.5,1.0}`, 10% d0 s42 | `runs/rq1/rq1-12tune_lam*_*/`; pilih λ dengan BACC4 **validasi** tertinggi, catat di adendum 3 + STATUS | smoke ✅ 2026-11-13 (15/15 bersama supcon12); menunggu commit + izin |
| RQ1-12/13 (SupCon + CE; negatif berbobot jarak grade, M219) | `SUPCON_W=<λ> setsid nohup bash scripts/run_rq1.sh supcon12 >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: supcon12_one()`: 12 `--supcon-weight $SUPCON_W`, 13 + `--supcon-alpha 0.25`; `train.py: supcon_step()`, `models.py: supcon_loss()` | `runs/rq1/rq1-1{2,3}_*/` | smoke ✅ 2026-11-13; 1 epoch 100% ±75 s |
| RQ1-08/10 SimCLR, BYOL (adendum 4): pre-training + 6 fine-tune per metode | `setsid nohup bash scripts/run_rq1.sh ssl0810 >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: pretrain(), ssl_arm()` → `ramscore.pretrain --method simclr\|byol --model resnet34 --amp --resume`, lalu `train.py --init-from runs/rq1_pretrain/<arm>_<method>_resnet34/encoder.pt` | `runs/rq1_pretrain/<name>/` (`encoder.pt`, `encoder.sha256`, `done.json`, `history.csv`), `runs/rq1/rq1-{08,10}_*/` | smoke ✅ 2026-11-13 (12/12, pre-training 1 epoch `--limit 300`) |
| RQ1-14/15 normalitas M197 + gate (adendum 5) | `setsid nohup bash scripts/run_rq1.sh norm1415 >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: norm_one()` → `ramscore.normality train` (sendi grade 0 train berlabel) lalu `ramscore.normality gate --normality <rq1-14> --classifier runs/rq0/rq0-01_*` | `runs/rq1/rq1-1{4,5}_*/` (`metrics.json`, `test_predictions.csv`, `scores_{val,test}.csv`, `cutpoints.json`, RQ1-14 `normality.pt`) | smoke ✅ 2026-11-13 (12/12) |
| RQ1-07 siapkan RSNA (adendum 6) | `uv run python -m ramscore.boneage prepare --zip ../data/external/rsna-bone-age/raw/rsna-bone-age.zip --out ../data/external/rsna-bone-age/png512` (zip diunduh lewat `uvx --from kaggle --with six --with python-dotenv kaggle datasets download kmader/rsna-bone-age -p data/external/rsna-bone-age/raw`; Kaggle sistem rusak) | `boneage.py: prepare(), pad_square()` | `png512/images/<id>.png` 512 px, `labels.csv`, `SHA256SUMS`; 12.611/12.611, 982 MB; zip dihapus (izin) | ✅ 2026-11-13 |
| RQ1-07 bone age: pre-training + 6 fine-tune | `setsid nohup bash scripts/run_rq1.sh boneage07 >> runs/rq1.log 2>&1 < /dev/null &` | `run_rq1.sh: boneage07(), finetunes()` → `ramscore.boneage pretrain` (resume dari checkpoint) → `train.py --init-from runs/rq1_pretrain/rq1-07_boneage_resnet34/encoder.pt` | `runs/rq1_pretrain/rq1-07_boneage_resnet34/`, `runs/rq1/rq1-07_*/` | smoke ✅ 2026-11-13 (6/6); 1 epoch 91 s |
| SKELEX (RQ1-18) unduh | `curl -sSL -o <f> https://huggingface.co/skhoha/SKELEX/resolve/main/<f>` untuk `config.json`, `preprocessor_config.json`, `model.safetensors` | — | `data/external/skelex/` + `SHA256SUMS` (ViT-L/16, 224, 3 kanal, `ViTMAEForPreTraining`; CC-BY-NC-ND-4.0) | ✅ 2026-11-13 |
| SKELEX → timm + verifikasi (adendum 7) | `CUDA_VISIBLE_DEVICES= uv run python -m ramscore.skelex convert --src ../data/external/skelex --out ../data/external/skelex/encoder_timm.pt`, lalu `... skelex verify ...` | `skelex.py: hf_to_timm(), reference_cls(), compare()` | `encoder_timm.pt` (294 tensor, 1,16 GB) + `.sha256`; verify: cosine 1,000000, error relatif 1,9e-4 | ✅ 2026-11-13 |
| RQ1-18a/b SKELEX LP + fine-tune | `setsid nohup bash scripts/run_rq1.sh skelex18 >> runs/rq1.log 2>&1 < /dev/null &` (setelah ukur 1 epoch) | `run_rq1.sh: skelex18_one()`: 18a `--freeze-backbone --lr 1e-2`; 18b `--layer-decay 0.85 --lr 1e-4 --warmup 0.1 --grad-checkpointing` | `runs/rq1/rq1-18{a,b}_*/` | smoke ✅ 2026-11-13 (`--limit 64`, batch 8: puncak 2,6 / 6,4 GB) |
| Antrean memakai salinan beku | semua tahap `run_rq1.sh <stage>` | `run_rq1.sh` awal: salin diri ke `runs/.queue_snapshots/` lalu `exec` | mengedit `run_rq1.sh` tidak memengaruhi antrean yang berjalan | ✅ 2026-11-13 |
| MoCo v3 / iBOT / MMD | 🛠 `--method simclr\|mocov3\|byol\|ibot`, … | 🛠 `ssl.py` + `pretrain.py` | 🛠 | 🛠 |
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
| RQ1: keluarga Holm F1–F4 (ukuran tetap), pilihan berbasis validasi, test − val | `uv run --group analysis python -m ramscore.rq1` | `rq1.py: selections()`, `select()`, `resolve()`, `family()`, `analyse()`, `report()` (memakai `e6abl.paired/contrast/add_holm`, `e7.prediction_stats`) | `runs/rq1_analysis/report.md` + `rq1.json`, disalin ke `reports/rq1.md` | ✅ 2026-11-12 (RQ1-01…06, 17a/b); ✅ 2026-11-13 + bagian sensitivitas adendum 2 (`sensitivity()`, `select(qwk_floor_arm=)`), kontras utama identik |
| RQ1: celah train − val checkpoint terpilih (GPU, inferensi saja) | `uv run --group analysis python -m ramscore.rq1 gap` | `rq1.py: gap()` → `e8.py: gap(runs=…)` | `runs/rq1_analysis/gap.json` (dibaca otomatis oleh laporan) | 🛠 belum dijalankan |
| Kontras ad hoc dari CLI | `uv run python -m ramscore.compare <baseline> <arm…> --pattern '../runs/rq<n>/{}_*_f1.0_*' --out <file>` | `compare.py: main()` | `<file>.json`, `<file>.md` | ✅ (kode lama) |

## 6. Flag dan modul baru (diisi saat dibuat)
| Tanggal | Flag / modul | File : fungsi | Dipakai oleh arm | Test |
|---|---|---|---|---|
| 2026-11-11 | modul `rq0.py` | `rq0.py: analyse(), reproduction(), run_summary(), verdict(), report()` | RQ0-01, RQ0-02 | `tests/test_rq0.py` (10) |
| 2026-11-11 | `--freeze-backbone` kini boleh dengan `--pretrained` (linear probe ImageNet); flag baru `--keep-head` (dengan `--init-from`: muat checkpoint lengkap termasuk kepala, tahap 2 LP-FT) | `train.py: check_init_args()`, `main()`; `models.py: load_full()` | RQ1-17a–d | `tests/test_rq1.py` (11); smoke LP → LP-FT: backbone LP = ImageNet persis, LP-FT mengubah backbone |
| 2026-11-12 | modul `rq1.py`; `e8.gap()` menerima `runs=` (default tidak berubah) | `rq1.py: select(), selections(), resolve(), family(), analyse(), gap(), report()`; `e8.py: gap()` | RQ0-01, RQ1-* | `tests/test_rq1_analysis.py` (7) |
| 2026-11-13 | `rq1.py`: pembanding sensitivitas adendum 2 (terbaik dengan val QWK ≥ RQ0-01) | `rq1.py: select(qwk_floor_arm=)`, `selections()`, `sensitivity()`, `selection_line()`, `fmt_sensitivity()` | kontras "− metode terbaik" (RQ1-13, 15, 16, 18b) | `tests/test_rq1_analysis.py` (10) |
| 2026-11-13 | `--supcon-weight`, `--supcon-temp` (0,1), `--supcon-alpha` (M219 α); dua view per sendi, CE hanya view pertama; kepala proyeksi 512→512→128 tidak disimpan | `train.py: check_supcon_args(), supcon_step()`; `models.py: supcon_loss(), ProjectionHead, pooled_and_logits()`; `data.py: BEDataset.views, to_tensor()` | RQ1-12, RQ1-13 | `tests/test_supcon.py` (19): rumus vs transkripsi manual, α = 1 = SupCon polos, anchor tanpa positif, dua view independen, kombinasi flag |
| 2026-11-13 | `pretrain.py --method simclr\|byol`, `--temperature`; penanda `done.json` di akhir pre-training (semua metode) | `ssl.py: SimCLR, BYOL, nt_xent(), byol_regression(), mlp()`; `pretrain.py: DEFAULTS, EMA, main()` | RQ1-08, RQ1-10 | `tests/test_ssl_rq1.py` (8): NT-Xent vs transkripsi manual, rentang loss BYOL, target tanpa gradien + EMA, encoder termuat ke classifier, default = resep E1 DINO |
| 2026-11-13 | modul `normality.py` (SS-FewSOME tahap 1 M197 + gate) | `normality.py: cutpaste(), FewSOMEPairs, PatchNet, fewsome_loss(), centre_of(), scores_of(), choose_cutpoints(), choose_gate(), train(), gate()` | RQ1-14, RQ1-15 | `tests/test_normality.py` (8): CutPaste hanya mengubah area mask, target patch, cut-point memulihkan skor terpisah, gate, pasangan tidak pernah dengan dirinya sendiri |
| 2026-11-13 | modul `boneage.py` (siapkan RSNA + pre-training regresi bone age M005) | `boneage.py: pad_square(), prepare(), split_ids(), BoneAgeDataset, pretrain()` | RQ1-07 | `tests/test_boneage.py` (3): pad di tengah, split ber-seed dan lengkap, tata letak zip Kaggle |
| 2026-11-13 | modul `skelex.py`; `train.py --grad-checkpointing` | `skelex.py: hf_to_timm(), reference_cls(), compare(), convert(), verify()`; `train.py: main()` (`net.set_grad_checkpointing`) | RQ1-18a/b | `tests/test_skelex.py` (3): ViT kecil hasil konversi = forward referensi, tanpa folding berbeda, checkpointing tidak mengubah output/gradien |
