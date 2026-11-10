"""
aug_hsv.py : HSV (renk tonu / doygunluk / parlaklık) augmentation scripti
=========================================================================

Ne yapar?
    Bir klasördeki görsel + YOLO etiket çiftlerini okur, her görselin
    renk tonunu (Hue), doygunluğunu (Saturation) ve parlaklığını (Value)
    küçük, rastgele miktarlarda değiştirir. Farklı saat, güneş ve hava
    koşullarını taklit eder. Sonuçları ayrı bir çıktı klasörüne yazar.
    Orijinal dosyalara dokunmaz.

Girdi klasör yapısı (--input):
    <input>/images/*.jpg|.jpeg|.png
    <input>/labels/*.txt            (YOLO formatı: sınıf x_center y_center w h)

Çıktı klasör yapısı (--output), yoksa otomatik oluşturulur:
    <output>/images/img123_hsv.jpg
    <output>/labels/img123_hsv.txt
    <output>/augmentation_log_hsv.csv   (her dosya için kullanılan değerler)

Kullanılan değerler :
    Hue         : -2 ile +2 arası tam sayı kayma (OpenCV ölçeği 0-179,
                  yani yaklaşık ±4 derece). Çok küçük tutulur ki kanyaşın
                  (Johnsongrass) yeşili ile silk'in sarımsı tonu karışmasın.
    Saturation  : mevcut değer x U(0.70, 1.30)   (çarpımsal)
    Value       : mevcut değer x U(0.75, 1.25)   (çarpımsal)

Etiket mantığı:
    HSV görüntünün SADECE rengini değiştirir; şekil, boyut ve konum aynı
    kalır. Bu yüzden kutular değişmez ve .txt dosyası AYNEN kopyalanır
    (yeni adla). Kutu silinmez, kutu hesabı yapılmaz.

Rastgelelik:
    Her görsel için rastgele sayı üreteci, DOSYA ADINDAN türetilen bir
    seed ile başlatılır (hashlib). Böylece script tekrar çalıştırılınca
    aynı görsel için aynı değerler çıkar. (Python'un hash() fonksiyonu
    kullanılmaz, çünkü her çalıştırmada farklı sonuç verir.)

Diğer scriptlerden (run_klasor1.py / run_klasor2.py) kullanmak için:
    from aug_hsv import augment, make_rng
    rng = make_rng(dosya_adi_uzantisiz)
    yeni_goruntu, kutular = augment(goruntu, kutular, rng)
"""

import argparse
import csv
import hashlib
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# AYARLAR
# --------------------------------------------------------------------------
SUFFIX = "_hsv"                       # çıktı dosya adına eklenen teknik eki
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 95
LOG_NAME = "augmentation_log_hsv.csv"  # her teknik kendi log dosyasını kullanır

HUE_SHIFT_MAX = 2                     # Hue: [-2, +2] tam sayı (OpenCV 0-179 ölçeği)
SAT_RANGE = (0.70, 1.30)              # Saturation çarpanı aralığı
VAL_RANGE = (0.75, 1.25)              # Value (parlaklık) çarpanı aralığı

# Üretilen görsel orijinalin neredeyse kopyası olmasın diye: saturation veya
# value çarpanlarından en az biri 1'den bu kadar uzak olmalı (ör. 0.10 = %10).
# Aksi halde çarpanlar yeniden çekilir (en fazla MAX_TRIES kez).
MIN_CHANGE = 0.10
MAX_TRIES = 20

# Kutu formatı (diğer scriptlerle aynı): (sinif, x_center, y_center, w, h)


# --------------------------------------------------------------------------
# 1) DOSYA OKUMA / YAZMA YARDIMCILARI
# --------------------------------------------------------------------------
def imread_unicode(path: Path):
    """Görseli okur (Windows'ta Türkçe/özel karakterli yollar için güvenli).
    Okunamazsa None döner."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path: Path, image) -> bool:
    """Görseli yazar. Başarılıysa True döner."""
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if ext in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def read_yolo_labels(label_path: Path):
    """YOLO .txt dosyasını okur ve doğrular.
    Dönüş: (kutular, gecerli_mi). Boş dosya geçerlidir (kutusuz görsel).
    Bozuk satır (5 değer değil, sayı değil, 0-1 dışı) varsa gecerli_mi=False."""
    boxes = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, True

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            return [], False
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            return [], False
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            return [], False
        boxes.append((cls, xc, yc, w, h))
    return boxes, True


# --------------------------------------------------------------------------
# 2) RASTGELELİK: dosya adından türetilen seed
# --------------------------------------------------------------------------
def make_rng(stem: str, base_seed: int = 0) -> random.Random:
    """Dosya adından (uzantısız) sabit bir seed üretir ve ona bağlı bir
    rastgele sayı üreteci döndürür. Aynı ad + aynı base_seed = aynı sonuç.
    base_seed'i değiştirirseniz aynı görsel için farklı değerler çıkar."""
    digest = hashlib.sha256(f"{base_seed}:{stem}{SUFFIX}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


# --------------------------------------------------------------------------
# 3) TEKNİĞİN ÇEKİRDEĞİ
# --------------------------------------------------------------------------
def sample_params(rng: random.Random):
    """Rastgele (hue_kaymasi, sat_carpani, val_carpani) üçlüsü seçer.
    Saturation ve value çarpanlarının ikisi de 1'e çok yakınsa (görsel
    neredeyse aynı kalırdı) yeniden çeker."""
    hue_shift = rng.randint(-HUE_SHIFT_MAX, HUE_SHIFT_MAX)
    sat_factor = val_factor = 1.0
    for _ in range(MAX_TRIES):
        sat_factor = rng.uniform(*SAT_RANGE)
        val_factor = rng.uniform(*VAL_RANGE)
        if max(abs(sat_factor - 1.0), abs(val_factor - 1.0)) >= MIN_CHANGE:
            break
    return hue_shift, sat_factor, val_factor


def apply_hsv(image, hue_shift: int, sat_factor: float, val_factor: float):
    """Görüntüyü BGR -> HSV'ye çevirir, kanalları değiştirir, BGR'ye döndürür.

    OpenCV (uint8) ölçekleri:  H: 0-179 (çember),  S: 0-255,  V: 0-255.
    - Hue döngüseldir: 179'un üstü 0'a sarar (mod 180), kırpılmaz.
    - Saturation ve Value çarpılır, 0-255 aralığına kırpılır.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h = ((h.astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    s = np.clip(np.rint(s.astype(np.float32) * sat_factor), 0, 255).astype(np.uint8)
    v = np.clip(np.rint(v.astype(np.float32) * val_factor), 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def augment(image, boxes, rng: random.Random):
    """Diğer scriptlerin import edeceği giriş noktası.
    Girdi : görüntü, kutu listesi ve rastgele sayı üreteci (make_rng ile)
    Çıktı : renkleri değiştirilmiş görüntü ve AYNI kutu listesi."""
    hue_shift, sat_factor, val_factor = sample_params(rng)
    return apply_hsv(image, hue_shift, sat_factor, val_factor), boxes


# --------------------------------------------------------------------------
# 4) TEK ÇİFTİ İŞLEME
# --------------------------------------------------------------------------
def process_pair(image_path: Path, labels_dir: Path, out_images: Path,
                 out_labels: Path, overwrite: bool, base_seed: int):
    """Bir görsel-etiket çiftini işler.
    Dönüş: (durum, kutu_sayisi, hue_kaymasi, sat_carpani, val_carpani).
    Durum: ok | etiket_yok | hatali_etiket | goruntu_okunamadi |
           cikti_var_atlandi | yazma_hatasi"""
    label_path = labels_dir / (image_path.stem + ".txt")
    out_image_path = out_images / (image_path.stem + SUFFIX + image_path.suffix.lower())
    out_label_path = out_labels / (image_path.stem + SUFFIX + ".txt")
    empty = (None, None, None)

    if not label_path.exists():
        return ("etiket_yok", 0, *empty)
    if not overwrite and (out_image_path.exists() or out_label_path.exists()):
        return ("cikti_var_atlandi", 0, *empty)

    boxes, valid = read_yolo_labels(label_path)   # sadece doğrulama için
    if not valid:
        return ("hatali_etiket", 0, *empty)

    image = imread_unicode(image_path)
    if image is None:
        return ("goruntu_okunamadi", 0, *empty)

    rng = make_rng(image_path.stem, base_seed)
    hue_shift, sat_factor, val_factor = sample_params(rng)
    new_image = apply_hsv(image, hue_shift, sat_factor, val_factor)

    # Görsel ve etiketi birlikte yaz; biri başarısız olursa yetim dosya kalmasın.
    if not imwrite_unicode(out_image_path, new_image):
        return ("yazma_hatasi", 0, *empty)
    try:
        shutil.copyfile(label_path, out_label_path)   # etiket AYNEN kopyalanır
    except OSError:
        out_image_path.unlink(missing_ok=True)
        return ("yazma_hatasi", 0, *empty)

    return ("ok", len(boxes), hue_shift, round(sat_factor, 3), round(val_factor, 3))


# --------------------------------------------------------------------------
# 5) KLASÖRÜ İŞLEME
# --------------------------------------------------------------------------
def process_folder(input_dir: Path, output_dir: Path, limit=None,
                   overwrite=False, base_seed=0):
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
            writer.writerow(["teknik", "gorsel", "durum", "kutu_sayisi",
                             "hue_kaymasi", "sat_carpani", "val_carpani"])

        for i, image_path in enumerate(image_paths, 1):
            status, n_boxes, hue, sat, val = process_pair(
                image_path, labels_dir, out_images, out_labels, overwrite, base_seed)
            writer.writerow(["hsv", image_path.name, status, n_boxes, hue, sat, val])
            counts[status] = counts.get(status, 0) + 1
            total_boxes += n_boxes
            if i % 500 == 0:
                print(f"  {i}/{len(image_paths)} işlendi...")

    print("\n=== ÖZET ===")
    for status, n in sorted(counts.items()):
        print(f"  {status:20s}: {n}")
    print(f"  Kopyalanan toplam kutu: {total_boxes}")
    print(f"  Log dosyası           : {log_path}")
    if set(counts) - {"ok"}:
        print("UYARI: 'ok' dışı durumlar var, log dosyasına bakın.")


# --------------------------------------------------------------------------
# 6) KOMUT SATIRI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HSV augmentation (YOLO)")
    parser.add_argument("--input", required=True,
                        help="İçinde images/ ve labels/ olan girdi klasörü")
    parser.add_argument("--output", required=True,
                        help="Çıktı klasörü (images/ ve labels/ burada oluşturulur)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sadece ilk N görseli işle (deneme için)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Çıktıda aynı adlı dosya varsa üzerine yaz")
    parser.add_argument("--seed", type=int, default=0,
                        help="Temel seed (değiştirirseniz aynı görsel için farklı değerler çıkar)")
    args = parser.parse_args()

    process_folder(Path(args.input), Path(args.output), args.limit,
                   args.overwrite, args.seed)


if __name__ == "__main__":
    main()