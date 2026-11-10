"""
run_klasor2.py : Klasör 2 (diğer görseller) TRAIN parçalarının her birine KENDİ tekniğini uygular.

    part1_flip     -> flip
    part2_scale    -> scale
    part3_rotation -> rotation
    part4_shear    -> shear
    part5_hsv      -> hsv
    part6_shadow   -> shadow
    part7_crop     -> crop

Her görsele sadece 1 teknik uygulanır: 7004 görsel -> en fazla 7004 yeni görsel.

Çıktı: augmented\\klasor_2\\images  ve  augmented\\klasor_2\\labels
(her tekniğin kendi log dosyası da çıktı klasörüne yazılır)

Kullanım:  python run_klasor2.py
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

TRAIN_DIR = ROOT / "Klasor_2(Diger)" / "train"   # içinde part1_flip, part2_scale ...
OUTPUT_DIR = ROOT / "augmented" / "klasor_2"

LIMIT = None   # Deneme için 20 yazın (her parça sadece ilk 20 görseli işler)

# (parça klasörü adı, o parçaya uygulanacak teknik)
PARTS = [
    ("part1_flip", aug_flip),
    ("part2_scale", aug_scale),
    ("part3_rotation", aug_rotation),
    ("part4_shear", aug_shear),
    ("part5_hsv", aug_hsv),
    ("part6_shadow", aug_shadow),
    ("part7_crop", aug_crop),
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

    for part_name, module in PARTS:
        part_dir = TRAIN_DIR / part_name
        n_input = count_files(part_dir / "images", IMAGE_EXTS)
        expected += min(n_input, LIMIT) if LIMIT else n_input

        print(f"########## {part_name} ({n_input} görsel) ##########")
        module.process_folder(part_dir, OUTPUT_DIR, limit=LIMIT)
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