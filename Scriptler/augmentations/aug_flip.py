"""
aug_flip.py : Yatay çevirme (horizontal flip) augmentation scripti
===================================================================

Ne yapar?
    Bir klasördeki görsel + YOLO etiket çiftlerini okur, her görseli yatay
    olarak aynalar (sol <-> sağ) ve etiketleri buna göre günceller.
    Sonuçları ayrı bir çıktı klasörüne yazar. Orijinal dosyalara dokunmaz.

Girdi klasör yapısı (--input):
    <input>/images/*.jpg|.jpeg|.png
    <input>/labels/*.txt            (YOLO formatı: sınıf x_center y_center w h)

Çıktı klasör yapısı (--output), yoksa otomatik oluşturulur:
    <output>/images/img123_flip.jpg
    <output>/labels/img123_flip.txt
    <output>/augmentation_log.csv   (her dosya için durum kaydı)

Diğer scriptlerden (run_klasor1.py / run_klasor2.py) kullanmak için:
    from aug_flip import augment
    yeni_goruntu, yeni_kutular = augment(goruntu, kutular)
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# AYARLAR
# --------------------------------------------------------------------------
SUFFIX = "_flip"                      # çıktı dosya adına eklenen teknik eki
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # taranacak görsel uzantıları
JPEG_QUALITY = 95                     # JPEG kayıt kalitesi
LOG_NAME = "augmentation_log_flip.csv"   # çıktı klasörüne yazılan kayıt dosyası

# Bir kutu şu formatta tutulur: (sinif, x_center, y_center, w, h)
# Değerler YOLO gibi normalize (0-1 arası), sinif ise int.


# --------------------------------------------------------------------------
# 1) DOSYA OKUMA / YAZMA YARDIMCILARI
# --------------------------------------------------------------------------
def imread_unicode(path: Path):
    """Görseli okur. cv2.imread Türkçe/özel karakterli yollarda Windows'ta
    sessizce başarısız olabildiği için np.fromfile + imdecode kullanılır.
    Okunamazsa None döner."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path: Path, image) -> bool:
    """Görseli yazar (imread_unicode'un karşılığı). Başarılıysa True döner."""
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if ext in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def read_yolo_labels(label_path: Path):
    """YOLO .txt dosyasını okur.

    Dönüş: (kutular, gecerli_mi)
        kutular   : [(sinif, x_center, y_center, w, h), ...]
        gecerli_mi: False ise dosyada bozuk bir satır var demektir
                    (5 değer değil, sayı değil, değerler 0-1 dışında...).
    Boş bir dosya geçerlidir (kutusuz arka plan görseli): ([], True).
    """
    boxes = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, True

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue                      # boş satırı atla
        if len(parts) != 5:
            return [], False              # beklenmeyen sütun sayısı
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            return [], False              # sayıya çevrilemedi
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            return [], False              # normalize aralığı dışında
        boxes.append((cls, xc, yc, w, h))
    return boxes, True


def write_yolo_labels(label_path: Path, boxes) -> None:
    """Kutuları YOLO formatında yazar (6 ondalık). Kutu yoksa boş dosya yazar."""
    lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --------------------------------------------------------------------------
# 2) TEKNİĞİN ÇEKİRDEĞİ: görüntüyü ve kutuları çevir
# --------------------------------------------------------------------------
def flip_image(image):
    """Görüntüyü yatay eksende aynalar (flipCode=1 -> sol/sağ)."""
    return cv2.flip(image, 1)


def flip_boxes(boxes):
    """Her kutunun x_center değerini 1 - x_center yapar. Geri kalan aynı."""
    return [(cls, 1.0 - xc, yc, w, h) for cls, xc, yc, w, h in boxes]


def augment(image, boxes):
    """Diğer scriptlerin import edeceği tek giriş noktası.
    Girdi : görüntü (numpy dizisi) ve kutu listesi
    Çıktı : çevrilmiş görüntü ve güncellenmiş kutu listesi
    """
    return flip_image(image), flip_boxes(boxes)


# --------------------------------------------------------------------------
# 3) TEK ÇİFTİ İŞLEME
# --------------------------------------------------------------------------
def process_pair(image_path: Path, labels_dir: Path, out_images: Path,
                 out_labels: Path, overwrite: bool):
    """Bir görsel-etiket çiftini işler.
    Dönüş: (durum, kutu_sayisi). Durum değerleri log'a yazılır:
        ok | etiket_yok | hatali_etiket | goruntu_okunamadi |
        cikti_var_atlandi | yazma_hatasi
    """
    label_path = labels_dir / (image_path.stem + ".txt")
    out_image_path = out_images / (image_path.stem + SUFFIX + image_path.suffix.lower())
    out_label_path = out_labels / (image_path.stem + SUFFIX + ".txt")

    # Önce hızlı kontroller: etiket var mı, çıktı zaten üretilmiş mi?
    if not label_path.exists():
        return "etiket_yok", 0
    if not overwrite and (out_image_path.exists() or out_label_path.exists()):
        return "cikti_var_atlandi", 0

    boxes, valid = read_yolo_labels(label_path)
    if not valid:
        return "hatali_etiket", 0

    image = imread_unicode(image_path)
    if image is None:
        return "goruntu_okunamadi", 0

    new_image, new_boxes = augment(image, boxes)

    # Görseli ve etiketi birlikte yaz; biri başarısız olursa ikisini de sil,
    # böylece çıktıda yetim (eşsiz) dosya kalmaz.
    if not imwrite_unicode(out_image_path, new_image):
        return "yazma_hatasi", 0
    try:
        write_yolo_labels(out_label_path, new_boxes)
    except OSError:
        out_image_path.unlink(missing_ok=True)
        return "yazma_hatasi", 0

    return "ok", len(new_boxes)


# --------------------------------------------------------------------------
# 4) KLASÖRÜ İŞLEME
# --------------------------------------------------------------------------
def process_folder(input_dir: Path, output_dir: Path, limit=None, overwrite=False):
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        sys.exit(f"HATA: '{input_dir}' içinde 'images' ve 'labels' klasörleri olmalı.")
    if input_dir.resolve() == output_dir.resolve():
        sys.exit("HATA: Girdi ve çıktı klasörü aynı olamaz (orijinaller korunmalı).")

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit:
        image_paths = image_paths[:limit]
    print(f"{len(image_paths)} görsel bulundu: {images_dir}")

    log_path = output_dir / LOG_NAME
    write_header = not log_path.exists()
    counts = {}
    total_boxes = 0

    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["teknik", "gorsel", "durum", "kutu_sayisi"])

        for i, image_path in enumerate(image_paths, 1):
            status, n_boxes = process_pair(image_path, labels_dir,
                                           out_images, out_labels, overwrite)
            writer.writerow(["flip", image_path.name, status, n_boxes])
            counts[status] = counts.get(status, 0) + 1
            total_boxes += n_boxes
            if i % 500 == 0:
                print(f"  {i}/{len(image_paths)} işlendi...")

    # Özet
    print("\n=== ÖZET ===")
    for status, n in sorted(counts.items()):
        print(f"  {status:20s}: {n}")
    print(f"  Yazılan toplam kutu : {total_boxes}")
    print(f"  Log dosyası         : {log_path}")
    if set(counts) - {"ok"}:
        print("UYARI: 'ok' dışı durumlar var, log dosyasına bakın.")


# --------------------------------------------------------------------------
# 5) KOMUT SATIRI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Yatay flip augmentation (YOLO)")
    parser.add_argument("--input", required=True,
                        help="İçinde images/ ve labels/ olan girdi klasörü")
    parser.add_argument("--output", required=True,
                        help="Çıktı klasörü (images/ ve labels/ burada oluşturulur)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sadece ilk N görseli işle (deneme için)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Çıktıda aynı adlı dosya varsa üzerine yaz")
    args = parser.parse_args()

    process_folder(Path(args.input), Path(args.output), args.limit, args.overwrite)


if __name__ == "__main__":
    main()