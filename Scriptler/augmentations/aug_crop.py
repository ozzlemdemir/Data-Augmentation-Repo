"""
aug_crop.py : Rastgele kırpma (random crop) augmentation scripti
================================================================

Ne yapar?
    Her görselden rastgele bir dikdörtgen bölge KESER ve ayrı bir görsel
    olarak kaydeder. Eğitimde görüntü yine aynı boyuta getirildiği için
    nesneler göreli olarak büyür; ayrıca nesnelerin kadraj içindeki konumu
    ve çevresi değişir. YOLO etiketlerini de kırpılan bölgeye göre günceller.
    Orijinal dosyalara dokunmaz.

Girdi klasör yapısı (--input):
    <input>/images/*.jpg|.jpeg|.png
    <input>/labels/*.txt            

Çıktı klasör yapısı (--output), yoksa otomatik oluşturulur:
    <output>/images/img123_crop.jpg
    <output>/labels/img123_crop.txt
    <output>/augmentation_log_crop.csv

Kullanılan değerler (sohbette kararlaştırılan):
    Kırpma boyutu  : genişlik ve yükseklik, her biri orijinalin %65-90'ı
                     (ayrı ayrı rastgele seçilir)
    Çözünürlük     : kırpılan parça orijinal çözünürlükte kaydedilir,
                     YENİDEN BÜYÜTÜLMEZ (kalite kaybı olmaz). Çıktı görsel
                     orijinalden küçük olur, bu normaldir.
    Kutu eşiği     : kutunun en az %60'ı kırpma içinde kalmalı, yoksa silinir
    Minimum kutu   : genişlik veya yükseklik 3 pikselden küçükse silinir
    Johnsongrass   : görselde Johnsongrass (sınıf 3) varsa, kırpma bölgesi
                     rastgele seçilen bir Johnsongrass kutusunu TAMAMEN
                     içerecek şekilde konumlandırılır (sığıyorsa).

Etiketler nasıl güncellenir? (3 adım)
    1. YOLO kutusu piksele çevrilir.
    2. Kırpma köşesi (x0, y0) çıkarılır, kutu kırpma sınırlarına kırpılır.
    3. Görünür oran %60'ın altındaysa veya kutu 3 pikselden küçükse silinir.
       Kalanlar kırpılan görüntünün boyutuna göre tekrar normalize edilir.
    (Kırpmada kutu şişmez; sadece kenarda kesilen kutular küçülür.)

Başarısızlık durumu:
    Kırpma sonrası hiç kutu kalmazsa (veya orijinalde Johnsongrass varken hiç
    Johnsongrass kutusu kalmazsa) farklı bir bölgeyle en fazla 20 kez yeniden
    denenir. Yine olmazsa görsel atlanır ve log'a yazılır.

Rastgelelik:
    Rastgele sayılar DOSYA ADINDAN türetilen seed ile üretilir (hashlib),
    script tekrar çalışınca aynı sonucu verir.

Diğer scriptlerden (run_klasor1.py / run_klasor2.py) kullanmak için:
    from aug_crop import augment, make_rng
    rng = make_rng(dosya_adi_uzantisiz)
    goruntu, kutular = augment(goruntu, kutular, rng)   # başarısızsa (None, None)
"""

import argparse
import csv
import hashlib
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# AYARLAR
# --------------------------------------------------------------------------
SUFFIX = "_crop"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 95
LOG_NAME = "augmentation_log_crop.csv"

CROP_FRAC = (0.65, 0.90)      # kırpmanın genişlik/yükseklik oranı aralığı
MIN_VISIBILITY = 0.60         # kutunun en az bu kadarı kırpma içinde kalmalı
MIN_BOX_PX = 3                # bundan küçük kutular silinir (piksel)
MAX_TRIES = 20                # uygun kırpma bulunamazsa en fazla kaç deneme
JOHNSONGRASS_CLASS = 3        # Klasör 2'de hiç yoktur; orada bu kural etkisizdir

# Kutu formatı (diğer scriptlerle aynı): (sinif, x_center, y_center, w, h)


# --------------------------------------------------------------------------
# 1) DOSYA OKUMA / YAZMA YARDIMCILARI
# --------------------------------------------------------------------------
def imread_unicode(path: Path):
    """Görseli okur (Windows'ta özel karakterli yollar için güvenli)."""
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
    Dönüş: (kutular, gecerli_mi). Boş dosya geçerlidir (kutusuz görsel)."""
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


def write_yolo_labels(label_path: Path, boxes) -> None:
    """Kutuları YOLO formatında yazar (6 ondalık)."""
    lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --------------------------------------------------------------------------
# 2) RASTGELELİK
# --------------------------------------------------------------------------
def make_rng(stem: str, base_seed: int = 0) -> random.Random:
    """Dosya adından sabit seed üretir. Aynı ad + aynı seed = aynı sonuç."""
    digest = hashlib.sha256(f"{base_seed}:{stem}{SUFFIX}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


# --------------------------------------------------------------------------
# 3) TEKNİĞİN ÇEKİRDEĞİ
# --------------------------------------------------------------------------
def yolo_to_pixel(box, w: int, h: int):
    """YOLO kutusunu piksel köşelerine (x1, y1, x2, y2) çevirir."""
    _, xc, yc, bw, bh = box
    return ((xc - bw / 2) * w, (yc - bh / 2) * h,
            (xc + bw / 2) * w, (yc + bh / 2) * h)


def sample_crop(rng: random.Random, w: int, h: int, anchor=None):
    """Kırpma bölgesini seçer. Dönüş: (x0, y0, kirpma_genisligi, kirpma_yuksekligi).

    anchor verilirse (bir Johnsongrass kutusunun piksel köşeleri), kırpma
    bölgesi o kutuyu tamamen içerecek şekilde konumlandırılır. Kutu kırpmaya
    sığmıyorsa konum tamamen rastgele seçilir."""
    cw = max(1, int(round(rng.uniform(*CROP_FRAC) * w)))
    ch = max(1, int(round(rng.uniform(*CROP_FRAC) * h)))

    # Önce tamamen rastgele konum
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)

    # Anchor varsa: kutuyu içine alan konum aralığından seç
    if anchor is not None:
        ax1, ay1, ax2, ay2 = anchor
        lo_x, hi_x = max(0, math.ceil(ax2 - cw)), min(w - cw, math.floor(ax1))
        lo_y, hi_y = max(0, math.ceil(ay2 - ch)), min(h - ch, math.floor(ay1))
        if lo_x <= hi_x:
            x0 = rng.randint(lo_x, hi_x)
        if lo_y <= hi_y:
            y0 = rng.randint(lo_y, hi_y)
    return x0, y0, cw, ch


def crop_image(image, x0: int, y0: int, cw: int, ch: int):
    """Görüntüden bölgeyi keser (yeniden boyutlandırma YOK)."""
    return image[y0:y0 + ch, x0:x0 + cw].copy()


def crop_boxes(boxes, x0: int, y0: int, cw: int, ch: int, w: int, h: int):
    """Kutuları kırpılan bölgeye göre günceller (yukarıdaki 3 adım).
    Eşiği geçemeyen kutular listeye alınmaz."""
    kept = []
    for box in boxes:
        cls = box[0]
        # 1) YOLO -> piksel, kırpma köşesini çıkar
        x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
        x1, x2 = x1 - x0, x2 - x0
        y1, y2 = y1 - y0, y2 - y0
        full_area = (x2 - x1) * (y2 - y1)
        if full_area <= 0:
            continue

        # 2) kırpma sınırlarına kırp
        cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
        cx2, cy2 = min(x2, float(cw)), min(y2, float(ch))
        if cx2 <= cx1 or cy2 <= cy1:
            continue                                   # tamamen dışarıda

        # 3) eşikler
        visible = ((cx2 - cx1) * (cy2 - cy1)) / full_area
        if visible < MIN_VISIBILITY:
            continue                                   # çok az görünüyor
        if (cx2 - cx1) < MIN_BOX_PX or (cy2 - cy1) < MIN_BOX_PX:
            continue                                   # çok küçük kaldı

        # piksel -> YOLO (kırpılan görüntünün boyutuna göre normalize)
        kept.append((cls,
                     (cx1 + cx2) / 2 / cw, (cy1 + cy2) / 2 / ch,
                     (cx2 - cx1) / cw, (cy2 - cy1) / ch))
    return kept


def result_is_acceptable(old_boxes, new_boxes) -> bool:
    """Kırpma sonucu kabul edilebilir mi?
    - Orijinalde kutu vardıysa en az bir kutu kalmalı.
    - Orijinalde Johnsongrass varsa en az bir Johnsongrass kalmalı."""
    if not old_boxes:
        return True                                    # kutusuz görsel: sorun yok
    if not new_boxes:
        return False
    had_jg = any(b[0] == JOHNSONGRASS_CLASS for b in old_boxes)
    has_jg = any(b[0] == JOHNSONGRASS_CLASS for b in new_boxes)
    return has_jg or not had_jg


def try_crop(image, boxes, rng: random.Random):
    """Kabul edilebilir bir kırpma bulana kadar (en fazla MAX_TRIES) dener.
    Dönüş: (yeni_goruntu, yeni_kutular, (x0, y0, cw, ch), deneme) veya None."""
    h, w = image.shape[:2]
    jg_boxes = [yolo_to_pixel(b, w, h) for b in boxes if b[0] == JOHNSONGRASS_CLASS]

    for attempt in range(1, MAX_TRIES + 1):
        anchor = rng.choice(jg_boxes) if jg_boxes else None
        x0, y0, cw, ch = sample_crop(rng, w, h, anchor)
        new_boxes = crop_boxes(boxes, x0, y0, cw, ch, w, h)
        if result_is_acceptable(boxes, new_boxes):
            return crop_image(image, x0, y0, cw, ch), new_boxes, (x0, y0, cw, ch), attempt
    return None


def augment(image, boxes, rng: random.Random):
    """Diğer scriptlerin kullanacağı giriş noktası.
    Dönüş: (kırpılmış görüntü, güncel kutular); başarısızsa (None, None)."""
    result = try_crop(image, boxes, rng)
    if result is None:
        return None, None
    return result[0], result[1]


# --------------------------------------------------------------------------
# 4) TEK ÇİFTİ İŞLEME
# --------------------------------------------------------------------------
def process_pair(image_path: Path, labels_dir: Path, out_images: Path,
                 out_labels: Path, overwrite: bool, base_seed: int):
    """Dönüş: (durum, kutu_oncesi, kutu_sonrasi, kirpma, deneme).
    kirpma = (x0, y0, genislik, yukseklik) veya None.
    Durum: ok | etiket_yok | hatali_etiket | goruntu_okunamadi |
           kutu_kalmadi | cikti_var_atlandi | yazma_hatasi"""
    label_path = labels_dir / (image_path.stem + ".txt")
    out_image_path = out_images / (image_path.stem + SUFFIX + image_path.suffix.lower())
    out_label_path = out_labels / (image_path.stem + SUFFIX + ".txt")

    if not label_path.exists():
        return "etiket_yok", 0, 0, None, None
    if not overwrite and (out_image_path.exists() or out_label_path.exists()):
        return "cikti_var_atlandi", 0, 0, None, None

    boxes, valid = read_yolo_labels(label_path)
    if not valid:
        return "hatali_etiket", 0, 0, None, None

    image = imread_unicode(image_path)
    if image is None:
        return "goruntu_okunamadi", len(boxes), 0, None, None

    result = try_crop(image, boxes, make_rng(image_path.stem, base_seed))
    if result is None:
        return "kutu_kalmadi", len(boxes), 0, None, None
    new_image, new_boxes, crop, attempt = result

    # Görsel ve etiketi birlikte yaz; biri başarısız olursa yetim dosya kalmasın.
    if not imwrite_unicode(out_image_path, new_image):
        return "yazma_hatasi", len(boxes), 0, None, None
    try:
        write_yolo_labels(out_label_path, new_boxes)
    except OSError:
        out_image_path.unlink(missing_ok=True)
        return "yazma_hatasi", len(boxes), 0, None, None

    return "ok", len(boxes), len(new_boxes), crop, attempt


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
    boxes_before = boxes_after = 0

    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["teknik", "gorsel", "durum", "kutu_oncesi",
                             "kutu_sonrasi", "x0", "y0", "kirpma_genislik",
                             "kirpma_yukseklik", "deneme"])
        for i, image_path in enumerate(image_paths, 1):
            status, n_old, n_new, crop, attempt = process_pair(
                image_path, labels_dir, out_images, out_labels, overwrite, base_seed)
            x0, y0, cw, ch = crop if crop else (None, None, None, None)
            writer.writerow(["crop", image_path.name, status, n_old, n_new,
                             x0, y0, cw, ch, attempt])
            counts[status] = counts.get(status, 0) + 1
            if status == "ok":
                boxes_before += n_old
                boxes_after += n_new
            if i % 500 == 0:
                print(f"  {i}/{len(image_paths)} işlendi...")

    print("\n=== ÖZET ===")
    for status, n in sorted(counts.items()):
        print(f"  {status:20s}: {n}")
    print(f"  Kutu (başarılı çiftlerde): {boxes_before} -> {boxes_after}")
    print(f"  Log dosyası              : {log_path}")
    if set(counts) - {"ok"}:
        print("UYARI: 'ok' dışı durumlar var, log dosyasına bakın.")


# --------------------------------------------------------------------------
# 6) KOMUT SATIRI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Random crop augmentation (YOLO)")
    parser.add_argument("--input", required=True,
                        help="İçinde images/ ve labels/ olan girdi klasörü")
    parser.add_argument("--output", required=True,
                        help="Çıktı klasörü (images/ ve labels/ burada oluşturulur)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sadece ilk N görseli işle (deneme için)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Çıktıda aynı adlı dosya varsa üzerine yaz")
    parser.add_argument("--seed", type=int, default=0,
                        help="Temel seed (değiştirirseniz farklı bölgeler çıkar)")
    args = parser.parse_args()
    process_folder(Path(args.input), Path(args.output), args.limit,
                   args.overwrite, args.seed)


if __name__ == "__main__":
    main()