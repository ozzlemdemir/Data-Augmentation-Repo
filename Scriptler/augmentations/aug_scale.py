"""
aug_scale.py : Ölçekleme (scale / yakınlaştırma-uzaklaştırma) augmentation scripti
===================================================================================

Ne yapar?
    Her görseli merkezinden yakınlaştırır (büyütür) veya uzaklaştırır
    (küçültür). Kamera mesafesi farkını taklit eder: uzak çekimlerde
    nesneler küçük, yakın çekimlerde büyük görünür. YOLO etiketlerini de
    aynı ölçeğe göre günceller. Orijinal dosyalara dokunmaz.

Girdi klasör yapısı (--input):
    <input>/images/*.jpg|.jpeg|.png
    <input>/labels/*.txt            (YOLO formatı: sınıf x_center y_center w h)

Çıktı klasör yapısı (--output), yoksa otomatik oluşturulur:
    <output>/images/img123_scale.jpg
    <output>/labels/img123_scale.txt
    <output>/augmentation_log_scale.csv

Kullanılan değerler (sohbette kararlaştırılan):
    Ölçek oranı    : 0.85 - 1.20 arası TEK oran (en = boy, yani şekil bozulmaz)
                     1.0'a çok yakın değerler (fark < 0.05) elenir, çünkü
                     görüntüyü neredeyse hiç değiştirmezdi.
    Merkez         : görüntünün ortası
    Kenar dolgusu  : yansıtma (BORDER_REFLECT_101), siyah boşluk kalmaz
                     (küçültmede kenarlar yansımayla dolar)
    Çıktı boyutu   : girdiyle aynı
    Kutu eşiği     : kutunun en az %60'ı görünür kalmalı, yoksa silinir
    Minimum kutu   : genişlik veya yükseklik 3 pikselden küçükse silinir

Büyütme (oran > 1) ve küçültme (oran < 1) arasındaki fark:
    - Büyütmede görüntünün kenar kısımları kadraj dışında kalır; kenardaki
      kutular kırpılır veya eşiği geçemezse silinir.
    - Küçültmede hiçbir kutu kadraj dışına çıkmaz, ama kutular küçülür
      (3 pikselden küçük kalanlar silinir).

Etiketler nasıl güncellenir? (4 adım)
    1. YOLO kutusu piksele çevrilir ve 4 köşesi alınır.
    2. 4 köşe, görüntüyü ölçekleyen AYNI matrisle taşınır.
    3. Köşeleri saran en küçük düz dikdörtgen yeni kutu olur (ölçeklemede
       kutu dikdörtgen kaldığı için şişme olmaz).
    4. Görüntü dışına taşan kısım kırpılır; görünür oran %60'ın altındaysa
       kutu silinir. Sonuç tekrar normalize edilip yazılır.

Başarısızlık durumu:
    Ölçekleme sonrası hiç kutu kalmazsa (veya orijinalde Johnsongrass varken
    hiç Johnsongrass kutusu kalmazsa) farklı bir oranla en fazla 20 kez
    yeniden denenir. Yine olmazsa görsel atlanır ve log'a yazılır.

Rastgelelik:
    Rastgele sayılar DOSYA ADINDAN türetilen seed ile üretilir (hashlib),
    script tekrar çalışınca aynı sonucu verir.

Diğer scriptlerden (run_klasor1.py / run_klasor2.py) kullanmak için:
    from aug_scale import augment, make_rng
    rng = make_rng(dosya_adi_uzantisiz)
    goruntu, kutular = augment(goruntu, kutular, rng)   # başarısızsa (None, None)
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# AYARLAR
# --------------------------------------------------------------------------
SUFFIX = "_scale"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 95
LOG_NAME = "augmentation_log_scale.csv"

SCALE_RANGE = (0.85, 1.20)    # ölçek oranı aralığı (1.0 = değişiklik yok)
MIN_CHANGE = 0.05             # oran 1.0'dan en az bu kadar uzak olmalı
MIN_VISIBILITY = 0.60         # kutunun en az bu kadarı görüntü içinde kalmalı
MIN_BOX_PX = 3                # bundan küçük kutular silinir (piksel)
MAX_TRIES = 20                # kutu kalmazsa en fazla kaç farklı oran denensin
JOHNSONGRASS_CLASS = 3        # orijinalde varsa ölçeklemeden sonra da kalmalı

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


def sample_scale(rng: random.Random) -> float:
    """Ölçek oranını seçer (0.85-1.20). 1.0'a çok yakın değerler elenir."""
    scale = 1.0
    for _ in range(MAX_TRIES):
        scale = rng.uniform(*SCALE_RANGE)
        if abs(scale - 1.0) >= MIN_CHANGE:
            break
    return scale


# --------------------------------------------------------------------------
# 3) TEKNİĞİN ÇEKİRDEĞİ
# --------------------------------------------------------------------------
def scale_matrix(w: int, h: int, scale: float):
    """Görüntü merkezi etrafında ölçekleyen 2x3 afin matrisi.
        x_yeni = scale * x + (1 - scale) * w/2
        y_yeni = scale * y + (1 - scale) * h/2
    Merkez noktası yerinde kalır. Hem görüntüye hem kutu köşelerine AYNI
    matris uygulanır; böylece etiket görüntüyle birlikte hareket eder."""
    return np.array([[scale, 0.0, (1.0 - scale) * w / 2.0],
                     [0.0, scale, (1.0 - scale) * h / 2.0]], dtype=np.float64)


def scale_image(image, matrix):
    """Görüntüyü ölçekler. Aynı boyutta kalır, boşluklar yansıtmayla dolar."""
    h, w = image.shape[:2]
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def scale_boxes(boxes, matrix, w: int, h: int):
    """Her YOLO kutusunu aynı matrisle ölçekler (yukarıdaki 4 adım).
    Eşiği geçemeyen kutular listeye alınmaz."""
    kept = []
    for cls, xc, yc, bw, bh in boxes:
        # 1) YOLO -> piksel köşeleri
        x1, x2 = (xc - bw / 2) * w, (xc + bw / 2) * w
        y1, y2 = (yc - bh / 2) * h, (yc + bh / 2) * h
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        # 2) köşeleri ölçekle
        scaled = corners @ matrix[:, :2].T + matrix[:, 2]

        # 3) köşeleri saran en küçük düz dikdörtgen
        nx1, ny1 = scaled.min(axis=0)
        nx2, ny2 = scaled.max(axis=0)
        full_area = (nx2 - nx1) * (ny2 - ny1)
        if full_area <= 0:
            continue

        # 4) görüntü sınırlarına kırp, görünür oranı hesapla
        cx1, cy1 = max(nx1, 0.0), max(ny1, 0.0)
        cx2, cy2 = min(nx2, float(w)), min(ny2, float(h))
        if cx2 <= cx1 or cy2 <= cy1:
            continue                                   # tamamen dışarıda
        visible = ((cx2 - cx1) * (cy2 - cy1)) / full_area
        if visible < MIN_VISIBILITY:
            continue                                   # çok az görünüyor
        if (cx2 - cx1) < MIN_BOX_PX or (cy2 - cy1) < MIN_BOX_PX:
            continue                                   # çok küçük kaldı

        # piksel -> YOLO (normalize)
        kept.append((cls,
                     (cx1 + cx2) / 2 / w, (cy1 + cy2) / 2 / h,
                     (cx2 - cx1) / w, (cy2 - cy1) / h))
    return kept


def result_is_acceptable(old_boxes, new_boxes) -> bool:
    """Ölçekleme sonucu kabul edilebilir mi?
    - Orijinalde kutu vardıysa en az bir kutu kalmalı.
    - Orijinalde Johnsongrass varsa en az bir Johnsongrass kalmalı."""
    if not old_boxes:
        return True                                    # kutusuz görsel: sorun yok
    if not new_boxes:
        return False
    had_jg = any(b[0] == JOHNSONGRASS_CLASS for b in old_boxes)
    has_jg = any(b[0] == JOHNSONGRASS_CLASS for b in new_boxes)
    return has_jg or not had_jg


def try_scale(image, boxes, rng: random.Random):
    """Kabul edilebilir bir sonuç bulana kadar (en fazla MAX_TRIES) dener.
    Dönüş: (yeni_goruntu, yeni_kutular, oran, deneme_sayisi) veya None."""
    h, w = image.shape[:2]
    for attempt in range(1, MAX_TRIES + 1):
        scale = sample_scale(rng)
        matrix = scale_matrix(w, h, scale)
        new_boxes = scale_boxes(boxes, matrix, w, h)
        if result_is_acceptable(boxes, new_boxes):
            return scale_image(image, matrix), new_boxes, scale, attempt
    return None


def augment(image, boxes, rng: random.Random):
    """Diğer scriptlerin kullanacağı giriş noktası.
    Dönüş: (ölçeklenmiş görüntü, güncel kutular); başarısızsa (None, None)."""
    result = try_scale(image, boxes, rng)
    if result is None:
        return None, None
    return result[0], result[1]


# --------------------------------------------------------------------------
# 4) TEK ÇİFTİ İŞLEME
# --------------------------------------------------------------------------
def process_pair(image_path: Path, labels_dir: Path, out_images: Path,
                 out_labels: Path, overwrite: bool, base_seed: int):
    """Dönüş: (durum, kutu_oncesi, kutu_sonrasi, oran, deneme).
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

    result = try_scale(image, boxes, make_rng(image_path.stem, base_seed))
    if result is None:
        return "kutu_kalmadi", len(boxes), 0, None, None
    new_image, new_boxes, scale, attempt = result

    # Görsel ve etiketi birlikte yaz; biri başarısız olursa yetim dosya kalmasın.
    if not imwrite_unicode(out_image_path, new_image):
        return "yazma_hatasi", len(boxes), 0, None, None
    try:
        write_yolo_labels(out_label_path, new_boxes)
    except OSError:
        out_image_path.unlink(missing_ok=True)
        return "yazma_hatasi", len(boxes), 0, None, None

    return "ok", len(boxes), len(new_boxes), round(scale, 4), attempt


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
                             "kutu_sonrasi", "olcek", "deneme"])
        for i, image_path in enumerate(image_paths, 1):
            status, n_old, n_new, scale, attempt = process_pair(
                image_path, labels_dir, out_images, out_labels, overwrite, base_seed)
            writer.writerow(["scale", image_path.name, status,
                             n_old, n_new, scale, attempt])
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
    parser = argparse.ArgumentParser(description="Scale augmentation (YOLO)")
    parser.add_argument("--input", required=True,
                        help="İçinde images/ ve labels/ olan girdi klasörü")
    parser.add_argument("--output", required=True,
                        help="Çıktı klasörü (images/ ve labels/ burada oluşturulur)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sadece ilk N görseli işle (deneme için)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Çıktıda aynı adlı dosya varsa üzerine yaz")
    parser.add_argument("--seed", type=int, default=0,
                        help="Temel seed (değiştirirseniz farklı oranlar çıkar)")
    args = parser.parse_args()
    process_folder(Path(args.input), Path(args.output), args.limit,
                   args.overwrite, args.seed)


if __name__ == "__main__":
    main()