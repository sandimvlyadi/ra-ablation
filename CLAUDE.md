# CLAUDE.md — ra-ablation

Baca file ini, lalu `STATUS.md`, sebelum mengerjakan apa pun. Pengguna menulis dalam bahasa Indonesia. Istilah teknis tetap dalam bahasa Inggris.

## Tujuan
Tesis S2 (ITB): grading erosi tulang per sendi (SvdH 0–5, dievaluasi pada 4 level 0/1/2/≥3) pada ROI sendi radiografi tangan RA.
- Data latih: **RAM-H1200**.
- Uji eksternal: **RheumaVIT**.

Bimbingan 2026-11-10 meminta ablasi beberapa metode per RQ, termasuk modifikasi metode yang ada. Ada dua kontribusi:
- **(1) Perbandingan sistematis** metode yang sudah ada.
- **(2) Modifikasi** yang diturunkan dari sifat data. Setiap modifikasi selalu dibandingkan dengan **dua** pembanding: versi aslinya dan metode terbaik yang sudah ada.

Rencana lengkap ada di `docs/PLAN_RQ.md`.

## Aturan kerja (wajib)
1. **Aturan analisis di-commit sebelum run pertama** setiap tahap (`rules/rq<n>_*.md`): arm, kontras, koreksi Holm, aturan putusan.
2. **Label val/test tidak pernah diubah.** Label correction (RQ5) hanya menyentuh label train. Prediksi out-of-fold dibuat di dalam train.
3. **Pemilihan checkpoint/hiperparameter hanya di validasi.** Test RAM-H1200 sudah dipakai di repo lama sejak E2, jadi semua hasil di sini *post hoc*. Nyatakan ini di setiap laporan. RheumaVIT adalah uji yang lebih bersih.
4. **Satu faktor per arm.** Kombinasi hanya di RQ2-13 (interaksi) dan RQ6.
5. **Tanya pengguna sebelum** commit, push, merge, meluncurkan antrean GPU, mengunduh file, atau menambah arm di luar `docs/PLAN_RQ.md`.
6. **Perkiraan waktu harus ditandai kasar.** Ukur 1 epoch sebelum arm yang mahal (ViT-L / SKELEX).
7. **PC bisa dimatikan pengguna.** Setiap antrean harus bisa di-resume, dan run yang sudah punya `metrics.json` dilewati.
8. **Perbarui `STATUS.md`** setiap selesai satu langkah, dan catat keputusan pengguna beserta tanggalnya. Untuk setiap perintah atau kode baru, tambahkan barisnya di `docs/RUNBOOK.md` (perintah → kode → output).
9. **Laporan untuk dosen deskriptif.** Hindari kode internal (B2, E8, dan sejenisnya) di bagian kesimpulan. Pakai nama metodenya.
10. Klaim "baru" untuk suatu modifikasi hanya setelah pencarian literatur yang terarah.

## Struktur
- `code/`: paket Python `ramscore` (disalin dari `slr-ra-experiments` @ b6e0135, lihat `code/SOURCE.txt`), `tests/`, dan `pyproject.toml`. Modul `e*.py` adalah warisan eksperimen lama dan dipakai ulang lewat fungsinya. Modul baru diberi nama `rq<n>.py`.
- `artifacts/encoders/`: encoder SSL hasil E1 (DINO/ChronoCon R34 & ViT-S, MAE ViT-S). Read-only, diverifikasi dengan `sha256sum -c artifacts/SHA256SUMS` (dijalankan dari `artifacts/`). Tidak masuk git.
- `data/`: `RAM-H1200-v1` dan `RAM-W600` adalah symlink ke `/data/Smart-X/tesis-fix/dataset/`. `derived/rheumavit` berisi ROI RheumaVIT (`manifest.csv`). Tidak masuk git.
- `rules/`: pra-spesifikasi per tahap. `scripts/`: antrean `run_rq<n>.sh`. `runs/rq<n>/`: output (tidak masuk git). `reports/`: laporan analisis yang di-commit.

## Perintah inti
- Lingkungan: `cd code && uv sync --group dev --group analysis && uv run pytest -q` (209 test lulus pada 2026-11-11).
- Path data ditulis absolut di kode (`train.py: DEFAULT_ROOT`) dan masih valid. Modul analisis lama berasumsi `runs/e4/rheumavit/manifest.csv`, jadi berikan `--manifest ../data/derived/rheumavit/manifest.csv`.
- Detail per langkah ada di `docs/RUNBOOK.md`.

## Lingkungan
- GPU: RTX 3060 12 GB (kadang dipakai aplikasi lain).
- Kaggle CLI sudah terpasang dengan `~/.kaggle/kaggle.json`, untuk dataset RSNA Bone Age (RQ1-07). Tetap tanya sebelum mengunduh.
- Remote: `git@github.com:sandimvlyadi/ra-ablation.git` (privat). SSH ke GitHub kadang gagal sekali (publickey). Mencoba ulang biasanya berhasil.
