"""
aug_shadow.py : Rastgele gölge (random shadow) augmentation scripti
===================================================================

Ne yapar?
    Bir klasördeki görsel + YOLO etiket çiftlerini okur, her görsele 1 veya
    2 adet yapay gölge ekler (bulut gölgesi, yaprak/nesne gölgesi gibi) ve
    sonucu ayrı bir çıktı klasörüne yazar. Orijinal dosyalara dokunmaz.

Girdi klasör yapısı (--input):
    <input>/images/*.jpg|.jpeg|.png
    <input>/labels/*.txt            (YOLO formatı: sınıf x_center y_center w h)

Çıktı klasör yapısı (--output), yoksa otomatik oluşturulur:
    <output>/images/img123_shadow.jpg
    <output>/labels/img123_shadow.txt
    <output>/augmentation_log_shadow.csv   (her dosya için gölge bilgisi)

Kullanılan değerler (sohbette kararlaştırılan):
    Gölge sayısı   : 1 veya 2 (rastgele)
    Yoğunluk       : her gölge için 0.20 - 0.40 (o bölgeyi %20-40 karartır)
    Bölge          : tüm görüntü (gölge herhangi bir yere düşebilir)
    Şekil          : yarısı eğik şerit (yaprak/nesne gölgesi), yarısı
                     elips (bulut gölgesi)

Gölge nasıl üretilir?
    1. Her gölge için rastgele bir şekil çizilir (şerit veya elips).
    2. Şekle rastgele bir yoğunluk (0.20-0.40) atanır.
    3. Birden fazla gölge varsa her pikselde EN KOYU olan geçerli olur
       (üst üste binince 0.40'ın ötesine karartılmaz).
    4. Gölgenin SADECE KENARLARI Gaussian ile yumuşatılır (gerçek gölgeler
       keskin kenarlı olmaz). Bulanıklık yalnızca gölge maskesine uygulanır,
       görüntünün kendisi bulanıklaştırılmaz.
    5. Piksel = piksel x (1 - karartma). Renk tonu korunur, sadece kararır.

Etiket mantığı:
    Gölge görüntünün şeklini/konumunu değiştirmez, sadece parlaklığı
    değiştirir. Bu yüzden kutular değişmez ve .txt dosyası AYNEN kopyalanır
    (yeni adla). Kutu silinmez, tekrar deneme döngüsü gerekmez.

Rastgelelik:
    Her görsel için rastgele sayı üreteci DOSYA ADINDAN türetilen bir seed
    ile başlatılır (hashlib). Script tekrar çalışınca aynı görsel için aynı
    gölge üretilir. (Python'un hash() fonksiyonu kullanılmaz.)

Diğer scriptlerden (run_klasor1.py / run_klasor2.py) kullanmak için:
    from aug_shadow import augment, make_rng
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
SUFFIX = "_shadow"                       # çıktı dosya adına eklenen teknik eki
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 95
LOG_NAME = "augmentation_log_shadow.csv"  # her teknik kendi log dosyasını kullanır

NUM_SHADOWS = (1, 2)                     # gölge sayısı aralığı (dahil)
INTENSITY_RANGE = (0.20, 0.40)           # her gölgenin yoğunluğu
STRIP_PROB = 0.5                         # gölgenin şerit olma olasılığı (yoksa elips)

# Şerit gölge: genişlik görüntü genişliğinin bu oranı kadar,
# alt kenarı üst kenara göre bu oranda yana kayabilir (eğik şerit).
STRIP_WIDTH_FRAC = (0.15, 0.40)
STRIP_SLANT_FRAC = (-0.30, 0.30)

# Elips (bulut) gölge: yarı eksenler görüntü genişlik/yüksekliğinin bu oranı.
BLOB_AXIS_FRAC = (0.15, 0.35)

# Gölge kenar yumuşatma: Gaussian sigma = bu oran x görüntünün kısa kenarı.
EDGE_SIGMA_FRAC = 0.015

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
    base_seed'i değiştirirseniz aynı görsel için farklı gölge çıkar."""
    digest = hashlib.sha256(f"{base_seed}:{stem}{SUFFIX}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


# --------------------------------------------------------------------------
# 3) GÖLGE ŞEKİLLERİ (her biri 0/1 değerli sert bir maske döndürür)
# --------------------------------------------------------------------------
def strip_mask(rng: random.Random, h: int, w: int):
    """Görüntüyü yukarıdan aşağıya kesen eğik bir şerit (yaprak/nesne gölgesi).
    Dörtgenin üst kenarı rastgele bir yerde, alt kenarı yana kaymış olabilir."""
    width = rng.uniform(*STRIP_WIDTH_FRAC) * w
    x_top = rng.uniform(0, max(1.0, w - width))
    x_bottom = x_top + rng.uniform(*STRIP_SLANT_FRAC) * w
    pts = np.array([[x_top, 0], [x_top + width, 0],
                    [x_bottom + width, h], [x_bottom, h]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask, f"serit({width / w:.2f})"


def blob_mask(rng: random.Random, h: int, w: int):
    """Rastgele konumda, rastgele döndürülmüş bir elips (bulut gölgesi)."""
    cx = rng.uniform(0.1, 0.9) * w
    cy = rng.uniform(0.1, 0.9) * h
    axis_x = rng.uniform(*BLOB_AXIS_FRAC) * w
    axis_y = rng.uniform(*BLOB_AXIS_FRAC) * h
    angle = rng.uniform(0, 180)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(axis_x), int(axis_y)),
                angle, 0, 360, 1, -1)
    return mask, f"elips({axis_x / w:.2f}x{axis_y / h:.2f})"


# --------------------------------------------------------------------------
# 4) TEKNİĞİN ÇEKİRDEĞİ
# --------------------------------------------------------------------------
def build_darkening_map(rng: random.Random, h: int, w: int):
    """Görüntü boyutunda bir 'karartma haritası' üretir (değerler 0 - 0.40).
    0 = gölge yok, 0.30 = o piksel %30 karartılacak.
    Dönüş: (harita, aciklama). aciklama log içindir (şekil ve yoğunluklar)."""
    n_shadows = rng.randint(*NUM_SHADOWS)
    darkening = np.zeros((h, w), dtype=np.float32)
    descriptions = []

    for _ in range(n_shadows):
        make_shape = strip_mask if rng.random() < STRIP_PROB else blob_mask
        mask, shape_desc = make_shape(rng, h, w)
        intensity = rng.uniform(*INTENSITY_RANGE)
        # Üst üste binen gölgelerde en koyu olan geçerli olur (toplanmaz).
        darkening = np.maximum(darkening, mask.astype(np.float32) * intensity)
        descriptions.append(f"{shape_desc}@{intensity:.2f}")

    # Sadece gölge kenarlarını yumuşat (görüntünün kendisi bulanıklaşmaz).
    sigma = EDGE_SIGMA_FRAC * min(h, w)
    darkening = cv2.GaussianBlur(darkening, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return darkening, "|".join(descriptions)


def apply_shadow(image, darkening):
    """Her pikseli (1 - karartma) ile çarpar. Üç renk kanalı aynı oranda
    karardığı için renk tonu korunur, sadece parlaklık düşer."""
    shaded = image.astype(np.float32) * (1.0 - darkening)[..., None]
    return np.clip(np.rint(shaded), 0, 255).astype(np.uint8)


def augment(image, boxes, rng: random.Random):
    """Diğer scriptlerin import edeceği giriş noktası.
    Girdi : görüntü, kutu listesi ve rastgele sayı üreteci (make_rng ile)
    Çıktı : gölge eklenmiş görüntü ve AYNI kutu listesi."""
    h, w = image.shape[:2]
    darkening, _ = build_darkening_map(rng, h, w)
    return apply_shadow(image, darkening), boxes


# --------------------------------------------------------------------------
# 5) TEK ÇİFTİ İŞLEME
# --------------------------------------------------------------------------
def process_pair(image_path: Path, labels_dir: Path, out_images: Path,
                 out_labels: Path, overwrite: bool, base_seed: int):
    """Bir görsel-etiket çiftini işler.
    Dönüş: (durum, kutu_sayisi, gölge_aciklamasi, ortalama_karartma).
    Durum: ok | etiket_yok | hatali_etiket | goruntu_okunamadi |
           cikti_var_atlandi | yazma_hatasi"""
    label_path = labels_dir / (image_path.stem + ".txt")
    out_image_path = out_images / (image_path.stem + SUFFIX + image_path.suffix.lower())
    out_label_path = out_labels / (image_path.stem + SUFFIX + ".txt")

    if not label_path.exists():
        return "etiket_yok", 0, None, None
    if not overwrite and (out_image_path.exists() or out_label_path.exists()):
        return "cikti_var_atlandi", 0, None, None

    boxes, valid = read_yolo_labels(label_path)   # sadece doğrulama için
    if not valid:
        return "hatali_etiket", 0, None, None

    image = imread_unicode(image_path)
    if image is None:
        return "goruntu_okunamadi", 0, None, None

    h, w = image.shape[:2]
    rng = make_rng(image_path.stem, base_seed)
    darkening, description = build_darkening_map(rng, h, w)
    new_image = apply_shadow(image, darkening)

    # Görsel ve etiketi birlikte yaz; biri başarısız olursa yetim dosya kalmasın.
    if not imwrite_unicode(out_image_path, new_image):
        return "yazma_hatasi", 0, None, None
    try:
        shutil.copyfile(label_path, out_label_path)   # etiket AYNEN kopyalanır
    except OSError:
        out_image_path.unlink(missing_ok=True)
        return "yazma_hatasi", 0, None, None

    return "ok", len(boxes), description, round(float(darkening.mean()), 4)


# --------------------------------------------------------------------------
# 6) KLASÖRÜ İŞLEME
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
                             "golgeler", "ort_karartma"])

        for i, image_path in enumerate(image_paths, 1):
            status, n_boxes, description, mean_dark = process_pair(
                image_path, labels_dir, out_images, out_labels, overwrite, base_seed)
            writer.writerow(["shadow", image_path.name, status, n_boxes,
                             description, mean_dark])
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
# 7) KOMUT SATIRI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Random shadow augmentation (YOLO)")
    parser.add_argument("--input", required=True,
                        help="İçinde images/ ve labels/ olan girdi klasörü")
    parser.add_argument("--output", required=True,
                        help="Çıktı klasörü (images/ ve labels/ burada oluşturulur)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sadece ilk N görseli işle (deneme için)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Çıktıda aynı adlı dosya varsa üzerine yaz")
    parser.add_argument("--seed", type=int, default=0,
                        help="Temel seed (değiştirirseniz aynı görsel için farklı gölge çıkar)")
    args = parser.parse_args()

    process_folder(Path(args.input), Path(args.output), args.limit,
                   args.overwrite, args.seed)


if __name__ == "__main__":
    main()