# ra-ablation

Ablasi metode untuk grading erosi tulang per sendi pada radiografi tangan rheumatoid arthritis.
- **Data latih:** RAM-H1200.
- **Uji ketahanan:** RheumaVIT.
- **Urutan kerja:** per RQ.

| RQ | Topik |
|---|---|
| RQ1 | SSL / pre-training |
| RQ2 | class imbalance dan ordinalitas |
| RQ3 | ketahanan lintas dataset |
| RQ4 | kalibrasi dan rujukan selektif |
| RQ5 | label noise / teacher-student |
| RQ6 | kombinasi terbaik |

Proyek ini lanjutan dari repo `slr-ra-experiments` (lihat `code/SOURCE.txt`). Arahnya dari bimbingan 2026-11-10 (`docs/sources/advisor_2026-11-10.md`) dan systematic review metode SLR-M (`docs/sources/slr-m_kandidat_ablasi.md`).

| File | Isi |
|---|---|
| `CLAUDE.md` | Konteks dan aturan kerja untuk Claude Code |
| `STATUS.md` | Posisi terakhir dan langkah berikutnya |
| `docs/RUNBOOK.md` | Setiap langkah: perintah → kode yang dipanggil → output |
| `docs/PLAN_RQ.md` | Daftar arm per RQ: metode asli, modifikasi, pembanding, perkiraan biaya |
| `docs/KONFIGURASI.md` | Konfigurasi teknis bersama |
| `docs/HASIL_LAMA.md` | Hasil lama yang relevan sebagai konteks |
