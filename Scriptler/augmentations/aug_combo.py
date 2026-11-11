"""
aug_combo.py : Bir teknik + HSV'yi ARDIŞIK uygular ve TEK çıktı üretir
=====================================================================

Ne yapar?
    Her görsel için önce seçilen tekniği (flip, scale, rotation, shear,
    shadow veya crop), sonra HSV'yi uygular. İki işlem de BELLEKTE yapılır:
    ara dosya, geçici klasör ve çift kayıt yoktur, diskte sadece nihai
    çıktı oluşur (depolama tasarrufu).

Örnek: img123.jpg + rotation  ->  img123_rot_hsv.jpg  ve  img123_rot_hsv.txt

Etiketler:
    - Teknik kutuları değiştirebilir (flip, scale, rotation, shear, crop);
      bu hesap ilgili aug_*.py dosyasındaki augment() fonksiyonundan gelir.
    - HSV kutuları değiştirmez, etiket olduğu gibi kalır.
    Yani nihai etiket, tekniğin ürettiği etikettir.

Başarısızlık:
    Teknik uygun sonuç bulamazsa (kutu kalmadıysa) görsel atlanır ve log'a
    yazılır; HSV hiç çalıştırılmaz.

Rastgelelik:
    Teknik ve HSV değerleri, tek başlarına çalıştırıldıklarındakiyle AYNI
    seed mantığıyla (dosya adından) üretilir. Yani img123 için rotation açısı
    aug_rotation.py ile aynı çıkar, HSV değerleri de aug_hsv.py ile aynıdır.

Kullanım (run_klasor2.py içinden):
    import aug_combo, aug_rotation
    aug_combo.process_folder(girdi_klasoru, cikti_klasoru, aug_rotation)

Girdi : <girdi>/images ve <girdi>/labels
Çıktı : <cikti>/images ve <cikti>/labels, log: augmentation_log_combo.csv
"""

import csv
import sys
from pathlib import Path

import aug_flip
import aug_hsv
# Ortak okuma/yazma fonksiyonlarını tekrar yazmamak için aug_flip'ten alıyoruz.
from aug_flip import imread_unicode, imwrite_unicode, read_yolo_labels, write_yolo_labels

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LOG_NAME = "augmentation_log_combo.csv"


def apply_technique(technique, image, boxes, stem, base_seed):
    """Seçilen tekniği uygular. Dönüş: (görüntü, kutular) veya (None, None).
    aug_flip rastgelelik kullanmadığı için onun augment() fonksiyonu farklı."""
    if technique is aug_flip:
        return technique.augment(image, boxes)
    rng = technique.make_rng(stem, base_seed)
    return technique.augment(image, boxes, rng)


def process_pair(image_path: Path, labels_dir: Path, out_images: Path,
                 out_labels: Path, technique, overwrite: bool, base_seed: int):
    """Bir çifti işler. Dönüş: (durum, kutu_oncesi, kutu_sonrasi).
    Durum: ok | etiket_yok | hatali_etiket | goruntu_okunamadi |
           kutu_kalmadi | cikti_var_atlandi | yazma_hatasi"""
    suffix = technique.SUFFIX + aug_hsv.SUFFIX               # ör. "_rot_hsv"
    label_path = labels_dir / (image_path.stem + ".txt")
    out_image_path = out_images / (image_path.stem + suffix + image_path.suffix.lower())
    out_label_path = out_labels / (image_path.stem + suffix + ".txt")

    if not label_path.exists():
        return "etiket_yok", 0, 0
    if not overwrite and (out_image_path.exists() or out_label_path.exists()):
        return "cikti_var_atlandi", 0, 0

    boxes, valid = read_yolo_labels(label_path)
    if not valid:
        return "hatali_etiket", 0, 0

    image = imread_unicode(image_path)
    if image is None:
        return "goruntu_okunamadi", len(boxes), 0

    # 1. adım: teknik (görüntü ve kutular değişebilir)
    image, new_boxes = apply_technique(technique, image, boxes, image_path.stem, base_seed)
    if image is None:
        return "kutu_kalmadi", len(boxes), 0

    # 2. adım: HSV (sadece renk değişir, kutular aynı kalır)
    hsv_rng = aug_hsv.make_rng(image_path.stem, base_seed)
    image, new_boxes = aug_hsv.augment(image, new_boxes, hsv_rng)

    # Yaz: görsel ve etiket birlikte, biri başarısız olursa yetim dosya kalmasın
    if not imwrite_unicode(out_image_path, image):
        return "yazma_hatasi", len(boxes), 0
    try:
        write_yolo_labels(out_label_path, new_boxes)
    except OSError:
        out_image_path.unlink(missing_ok=True)
        return "yazma_hatasi", len(boxes), 0

    return "ok", len(boxes), len(new_boxes)


def process_folder(input_dir: Path, output_dir: Path, technique,
                   limit=None, overwrite=False, base_seed=0):
    """input_dir içindeki tüm çiftlere 'technique + HSV' uygular."""
    images_dir, labels_dir = input_dir / "images", input_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        sys.exit(f"HATA: '{input_dir}' içinde 'images' ve 'labels' klasörleri olmalı.")
    if input_dir.resolve() == output_dir.resolve():
        sys.exit("HATA: Girdi ve çıktı klasörü aynı olamaz (orijinaller korunmalı).")

    out_images, out_labels = output_dir / "images", output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit:
        image_paths = image_paths[:limit]
    name = technique.SUFFIX[1:] + "+hsv"
    print(f"{len(image_paths)} görsel bulundu ({name}): {images_dir}")

    log_path = output_dir / LOG_NAME
    write_header = not log_path.exists()
    counts = {}

    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(["teknik", "gorsel", "durum", "kutu_oncesi", "kutu_sonrasi"])
        for i, image_path in enumerate(image_paths, 1):
            status, n_old, n_new = process_pair(
                image_path, labels_dir, out_images, out_labels,
                technique, overwrite, base_seed)
            writer.writerow([name, image_path.name, status, n_old, n_new])
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(image_paths)} işlendi...")

    print("=== ÖZET ===")
    for status, n in sorted(counts.items()):
        print(f"  {status:20s}: {n}")
    if set(counts) - {"ok"}:
        print("UYARI: 'ok' dışı durumlar var, log dosyasına bakın.")