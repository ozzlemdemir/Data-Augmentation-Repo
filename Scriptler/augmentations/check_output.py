"""
check_output.py : Augmentation çıktılarını kontrol eder
=======================================================

Her kontrol edilecek klasör için 4 şey yapar:

  1. SAYI KONTROLÜ     : Her görselin eşleşen bir .txt dosyası var mı (ve tersi)?
  2. ETİKET KONTROLÜ   : Etiket dosyaları geçerli mi? (5 değer, sınıf 0-3,
                         değerler 0-1 arası, kutu görüntü dışına taşmıyor mu?)
                         Sınıf başına etiket sayısını da yazdırır.
  3. TEKNİK DAĞILIMI   : Dosya adındaki ekten (_flip, _scale, ...) hangi
                         teknikten kaç görsel olduğunu sayar.
  4. KUTU ÇİZDİRME     : Her teknikten rastgele N görseli seçer, .txt
                         dosyasındaki kutuları görselin ÜZERİNE çizer ve
                         önizleme olarak kaydeder. Kutuların nesnelerin
                         üzerinde durup durmadığını gözle kontrol edin.

Ek kontrol (val/test için): Val ve test klasörlerinde augmentation eki olan
dosya OLMAMALI. NO_AUGMENTED_FOLDERS listesine eklediğiniz klasörleri bu
yönden tarar.

Kutu renkleri:  Male = mavi, Female = kırmızı, Silk = sarı, Johnsongrass = yeşil

Kullanım:  python check_output.py
Önizlemeler: <ROOT>\\check_preview\\<klasor_adi>\\<teknik>\\...jpg
"""

import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# AYARLAR (sadece burayı düzenleyin)
# --------------------------------------------------------------------------
ROOT = Path(r"C:\Users\Lenovo\Desktop\Data-Augmentation-2026")

# İçinde images/ ve labels/ olan, kontrol edilecek klasörler.
# Birleştirme bitince şunları da ekleyebilirsiniz:  ROOT / "final_dataset" / "train"
CHECK_FOLDERS = [
    ROOT / "augmented_data" / "klasor_1",
    ROOT / "augmented_data" / "klasor_2",
]

# Augmentation eki İÇERMEMESİ gereken klasörler (val ve test). Boş bırakılabilir.
# Örnek:  ROOT / "final_dataset" / "val",  ROOT / "final_dataset" / "test"
NO_AUGMENTED_FOLDERS = []

PREVIEW_DIR = ROOT / "check_preview"   # kutu çizili önizlemelerin yazılacağı yer
SAMPLES_PER_TECHNIQUE = 10             # her teknikten kaç örnek çizdirilsin
PREVIEW_MAX_SIDE = 1600                # önizleme görselinin en uzun kenarı (piksel)
SEED = 0                               # aynı seed = aynı örnekler seçilir

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Uzun (birleşik) ekler ÖNCE gelmeli, yoksa "_rot_hsv" dosyası sadece "hsv" sayılır.
SUFFIXES = ["_flip_hsv", "_scale_hsv", "_rot_hsv", "_shear_hsv", "_shadow_hsv", "_crop_hsv",
            "_flip", "_scale", "_rot", "_shear", "_hsv", "_shadow", "_crop"]

CLASS_NAMES = ["Male", "Female", "Silk", "Johnsongrass"]        # sınıf 0, 1, 2, 3
CLASS_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (0, 200, 0)]  # BGR renkler

BOX_TOLERANCE = 1e-4   # kutunun görüntü sınırını aşmasına izin verilen küçük pay


# --------------------------------------------------------------------------
# 1) DOSYA YARDIMCILARI
# --------------------------------------------------------------------------
def imread_unicode(path: Path):
    """Görseli okur (Windows'ta özel karakterli yollar için güvenli)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path: Path, image) -> bool:
    """Görseli JPEG olarak yazar."""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buffer.tofile(str(path))
    return ok


def list_images(images_dir: Path):
    """images klasöründeki görsel dosyalarını sıralı döndürür."""
    return sorted(p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def technique_of(stem: str) -> str:
    """Dosya adındaki ekten tekniği bulur. Ek yoksa 'orijinal' döner."""
    for suffix in SUFFIXES:
        if stem.endswith(suffix):
            return suffix[1:]
    return "orijinal"


# --------------------------------------------------------------------------
# 2) ETİKET OKUMA VE DOĞRULAMA
# --------------------------------------------------------------------------
def parse_label_file(label_path: Path):
    """Etiket dosyasını okur. Dönüş: (kutular, hatalar).
    kutular: [(sinif, xc, yc, w, h), ...]   hatalar: ['satır 3: ...', ...]"""
    boxes, errors = [], []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, errors                      # boş dosya geçerlidir

    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"satır {line_no}: 5 değer olmalı, {len(parts)} var")
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"satır {line_no}: sayıya çevrilemedi")
            continue

        if not 0 <= cls < len(CLASS_NAMES):
            errors.append(f"satır {line_no}: geçersiz sınıf {cls}")
        elif not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"satır {line_no}: değerler 0-1 aralığı dışında")
        elif (xc - w / 2 < -BOX_TOLERANCE or xc + w / 2 > 1 + BOX_TOLERANCE or
              yc - h / 2 < -BOX_TOLERANCE or yc + h / 2 > 1 + BOX_TOLERANCE):
            errors.append(f"satır {line_no}: kutu görüntü dışına taşıyor")
        else:
            boxes.append((cls, xc, yc, w, h))
    return boxes, errors


# --------------------------------------------------------------------------
# 3) KUTU ÇİZDİRME
# --------------------------------------------------------------------------
def draw_boxes(image, boxes):
    """Kutuları görselin üzerine çizer (sınıf renginde dikdörtgen + sınıf adı)."""
    h, w = image.shape[:2]
    thickness = max(2, int(min(h, w) / 400))
    font_scale = max(0.5, min(h, w) / 1200)

    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        color = CLASS_COLORS[cls]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, CLASS_NAMES[cls], (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image


def make_previews(images_dir: Path, labels_dir: Path, preview_dir: Path, rng):
    """Her teknikten SAMPLES_PER_TECHNIQUE kadar rastgele görsel seçip
    kutuları çizerek preview_dir altına kaydeder."""
    by_technique = {}
    for image_path in list_images(images_dir):
        by_technique.setdefault(technique_of(image_path.stem), []).append(image_path)

    saved = 0
    for technique, paths in sorted(by_technique.items()):
        chosen = rng.sample(paths, min(SAMPLES_PER_TECHNIQUE, len(paths)))
        out_dir = preview_dir / technique
        out_dir.mkdir(parents=True, exist_ok=True)

        for image_path in chosen:
            label_path = labels_dir / (image_path.stem + ".txt")
            image = imread_unicode(image_path)
            if image is None or not label_path.exists():
                continue
            boxes, _ = parse_label_file(label_path)
            image = draw_boxes(image, boxes)

            # Önizlemeyi küçült (dosya boyutu makul kalsın)
            scale = PREVIEW_MAX_SIDE / max(image.shape[:2])
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            if imwrite_unicode(out_dir / (image_path.stem + ".jpg"), image):
                saved += 1
    return saved


# --------------------------------------------------------------------------
# 4) BİR KLASÖRÜ KONTROL ET
# --------------------------------------------------------------------------
def check_folder(folder: Path, rng, expect_no_augmented: bool = False):
    """Tek klasör için sayı, etiket, teknik dağılımı kontrolü ve önizleme."""
    print("=" * 60)
    print(f"KLASÖR: {folder}")
    images_dir, labels_dir = folder / "images", folder / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print("  HATA: images/ ve labels/ klasörleri bulunamadı, atlanıyor.")
        return

    image_paths = list_images(images_dir)
    label_paths = sorted(labels_dir.glob("*.txt"))
    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}

    # --- 1) Sayı kontrolü ---
    print(f"\n[1] SAYI KONTROLÜ")
    print(f"  Görsel: {len(image_paths)}   Etiket (.txt): {len(label_paths)}")
    no_label = sorted(image_stems - label_stems)
    no_image = sorted(label_stems - image_stems)
    if not no_label and not no_image:
        print("  OK: Her görselin etiketi, her etiketin görseli var.")
    if no_label:
        print(f"  UYARI: Etiketi olmayan {len(no_label)} görsel. Örnek: {no_label[:3]}")
    if no_image:
        print(f"  UYARI: Görseli olmayan {len(no_image)} etiket. Örnek: {no_image[:3]}")

    # --- 2) Etiket kontrolü ---
    print(f"\n[2] ETİKET KONTROLÜ")
    class_counts, bad_files, empty_files = Counter(), [], 0
    for label_path in label_paths:
        boxes, errors = parse_label_file(label_path)
        if errors:
            bad_files.append((label_path.name, errors[0]))
        if not boxes and not errors:
            empty_files += 1
        class_counts.update(b[0] for b in boxes)

    if not bad_files:
        print("  OK: Tüm etiket dosyaları geçerli.")
    else:
        print(f"  UYARI: {len(bad_files)} dosyada hata var. İlk 5 örnek:")
        for name, error in bad_files[:5]:
            print(f"    {name} -> {error}")
    print(f"  Boş etiket dosyası (kutusuz görsel): {empty_files}")
    print("  Sınıf başına etiket sayısı:")
    for cls, name in enumerate(CLASS_NAMES):
        print(f"    {cls} {name:13s}: {class_counts.get(cls, 0)}")

    # --- 3) Teknik dağılımı ---
    print(f"\n[3] TEKNİK DAĞILIMI (dosya adı ekine göre)")
    techniques = Counter(technique_of(p.stem) for p in image_paths)
    for technique, count in sorted(techniques.items()):
        print(f"    {technique:10s}: {count}")
    if expect_no_augmented:
        n_aug = sum(c for t, c in techniques.items() if t != "orijinal")
        if n_aug:
            print(f"  UYARI: Bu klasörde augmentation eki olan {n_aug} dosya var! "
                  f"(val/test'te olmamalı)")
        else:
            print("  OK: Augmentation eki olan dosya yok.")

    # --- 4) Kutu çizdirme ---
    if not expect_no_augmented:
        print(f"\n[4] KUTU ÇİZDİRME")
        saved = make_previews(images_dir, labels_dir, PREVIEW_DIR / folder.name, rng)
        print(f"  {saved} önizleme kaydedildi: {PREVIEW_DIR / folder.name}")
        print("  -> Önizlemeleri açıp kutuların nesnelerin üzerinde durduğunu kontrol edin.")


# --------------------------------------------------------------------------
# 5) ANA AKIŞ
# --------------------------------------------------------------------------
def main():
    rng = random.Random(SEED)
    for folder in CHECK_FOLDERS:
        check_folder(folder, rng)
    for folder in NO_AUGMENTED_FOLDERS:
        check_folder(folder, rng, expect_no_augmented=True)
    print("=" * 60)
    print("Kontrol bitti. 'UYARI' ve 'HATA' satırlarına bakın.")


if __name__ == "__main__":
    sys.exit(main())