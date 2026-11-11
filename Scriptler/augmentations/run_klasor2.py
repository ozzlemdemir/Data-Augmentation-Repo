"""
run_klasor2.py : Klasör 2 (diğer görseller) TRAIN parçalarına kendi tekniğini + HSV'yi uygular.

    part1_flip_and_hsv     -> flip    + HSV   (çıktı: img_flip_hsv.jpg)
    part2_scale_and_hsv    -> scale   + HSV   (çıktı: img_scale_hsv.jpg)
    part3_rotation_and_hsv -> rotation + HSV   (çıktı: img_rot_hsv.jpg)
    part4_shear_and_hsv    -> shear   + HSV   (çıktı: img_shear_hsv.jpg)
    part5_hsv              -> sadece HSV      (çıktı: img_hsv.jpg)
    part6_shadow_and_hsv   -> shadow  + HSV   (çıktı: img_shadow_hsv.jpg)
    part7_crop_and_hsv     -> crop    + HSV   (çıktı: img_crop_hsv.jpg)

Her görselden TEK çıktı üretilir (7004 görsel -> en fazla 7004 yeni görsel).
Teknik ve HSV bellekte ardışık uygulanır (aug_combo.py), ara dosya/geçici klasör yoktur.

Çıktı: augmented\\klasor_2\\images  ve  augmented\\klasor_2\\labels

Kullanım:  python run_klasor2.py
Bu dosya, aug_*.py dosyalarıyla AYNI klasörde (Scriptler) durmalıdır.
"""

from pathlib import Path

import aug_flip
import aug_scale
import aug_rotation
import aug_shear
import aug_hsv
import aug_shadow
import aug_crop
import aug_combo

# --------------------------------------------------------------------------
# AYARLAR (sadece burayı kontrol edin)
# --------------------------------------------------------------------------
ROOT = Path(r"C:\Users\Lenovo\Desktop\Data-Augmentation-2026")

TRAIN_DIR = ROOT / "Klasor_2(Diger)" / "train"   # içinde part1_flip_and_hsv, part2_scale_and_hsv ...
OUTPUT_DIR = ROOT / "augmented_data" / "klasor_2"

LIMIT = None   # Deneme için 20 yazın (her parça sadece ilk 20 görseli işler)

# (parça klasörü adı, teknik, HSV de eklensin mi?)
# Klasör adları bilgisayarınızdaki adlarla birebir aynı olmalı.
PARTS = [
    ("part1_flip_and_hsv", aug_flip, True),
    ("part2_scale_and_hsv", aug_scale, True),
    ("part3_rotation_and_hsv", aug_rotation, True),
    ("part4_shear_and_hsv", aug_shear, True),
    ("part5_hsv", aug_hsv, False),      # zaten sadece HSV
    ("part6_shadow_and_hsv", aug_shadow, True),
    ("part7_crop_and_hsv", aug_crop, True),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_files(folder: Path, extensions=None) -> int:
    """Klasördeki dosya sayısı (extensions verilirse sadece o uzantılar)."""
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and (extensions is None or p.suffix.lower() in extensions))


def main():
    expected = 0

    for part_name, technique, with_hsv in PARTS:
        part_dir = TRAIN_DIR / part_name
        n_input = count_files(part_dir / "images", IMAGE_EXTS)
        expected += min(n_input, LIMIT) if LIMIT else n_input

        print(f"########## {part_name} ({n_input} görsel) ##########")
        if with_hsv:
            aug_combo.process_folder(part_dir, OUTPUT_DIR, technique, limit=LIMIT)
        else:
            technique.process_folder(part_dir, OUTPUT_DIR, limit=LIMIT)
        print()

    # Sonuç kontrolü
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