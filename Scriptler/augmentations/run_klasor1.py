"""
run_klasor1.py : Klasör 1 (Johnsongrass'lı) TRAIN verisine 7 tekniği sırayla uygular.

Her teknik, train klasöründeki TÜM görsellere ayrı ayrı uygulanır:
    322 görsel x 7 teknik = en fazla 2254 yeni görsel (+ etiketleri)

Çıktı: augmented\\klasor_1\\images  ve  augmented\\klasor_1\\labels
(her tekniğin kendi log dosyası da çıktı klasörüne yazılır)

Kullanım:  python run_klasor1.py
"""

from pathlib import Path

import aug_flip
import aug_scale
import aug_rotation
import aug_shear
import aug_hsv
import aug_shadow
import aug_crop

# --------------------------------------------------------------------------
# AYARLAR (sadece burayı kontrol edin)
# --------------------------------------------------------------------------
ROOT = Path(r"C:\Users\Lenovo\Desktop\Data-Augmentation-2026")

# DİKKAT: Klasör adı bilgisayarınızdaki adla birebir aynı olmalı.
INPUT_DIR = ROOT / "Klasor_1(Johsongrassli)" / "train"   # içinde images/ ve labels/
OUTPUT_DIR = ROOT / "augmented" / "klasor_1"

LIMIT = None   # Deneme için 20 yazın (her teknik sadece ilk 20 görseli işler)

# Uygulanacak teknikler (sıra önemli değil)
TECHNIQUES = [
    ("flip", aug_flip),
    ("scale", aug_scale),
    ("rotation", aug_rotation),
    ("shear", aug_shear),
    ("hsv", aug_hsv),
    ("shadow", aug_shadow),
    ("crop", aug_crop),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_files(folder: Path, extensions=None) -> int:
    """Klasördeki dosya sayısı (extensions verilirse sadece o uzantılar)."""
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and (extensions is None or p.suffix.lower() in extensions))


def main():
    n_input = count_files(INPUT_DIR / "images", IMAGE_EXTS)
    n_used = min(n_input, LIMIT) if LIMIT else n_input
    print(f"Girdi: {INPUT_DIR}  ({n_input} görsel)\n")

    for name, module in TECHNIQUES:
        print(f"########## {name.upper()} ##########")
        module.process_folder(INPUT_DIR, OUTPUT_DIR, limit=LIMIT)
        print()

    # Sonuç kontrolü
    expected = n_used * len(TECHNIQUES)
    made_images = count_files(OUTPUT_DIR / "images", IMAGE_EXTS)
    made_labels = count_files(OUTPUT_DIR / "labels", {".txt"})
    print("=" * 50)
    print(f"Beklenen görsel : {expected}")
    print(f"Üretilen görsel : {made_images}")
    print(f"Üretilen etiket : {made_labels}")
    if made_images != made_labels:
        print("UYARI: Görsel ve etiket sayısı eşit değil!")
    if made_images != expected:
        print("UYARI: Beklenenden farklı. Log dosyalarında 'ok' dışı durumlara bakın.")


if __name__ == "__main__":
    main()