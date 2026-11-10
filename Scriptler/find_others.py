from __future__ import annotations

import random
import shutil
from pathlib import Path


ROOT = Path(r"C:\Users\Lenovo\Desktop\Data-Augmentation-2026")
SOURCE_DIR = ROOT / "orijinal_veriler_2026"
DESTINATION_DIR = ROOT / "Klasor_2(Diger)"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
EXPECTED_IMAGE_COUNT = 8755
PART_DIRECTORIES = (
    "part1_flip",
    "part2_scale",
    "part3_rotation",
    "part4_shear",
    "part5_hsv",
    "part6_shadow",
    "part7_crop",
)


def has_class_three(label_path: Path) -> bool:
    with label_path.open("r", encoding="utf-8-sig") as label_file:
        for line in label_file:
            columns = line.split()
            if columns and columns[0] == "3":
                return True
    return False


def find_image(label_path: Path) -> Path | None:
    matches = [
        label_path.with_suffix(extension)
        for extension in IMAGE_EXTENSIONS
        if label_path.with_suffix(extension).is_file()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Birden fazla görsel bulundu: {label_path.stem}")
    return matches[0] if matches else None


def copy_pairs(pairs: list[tuple[Path, Path]], destination: Path) -> None:
    image_directory = destination / "images"
    label_directory = destination / "labels"
    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_directory / image_path.name)
        shutil.copy2(label_path, label_directory / label_path.name)


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Kaynak klasör bulunamadı: {SOURCE_DIR}")

    eligible_pairs: list[tuple[Path, Path]] = []
    missing_image_count = 0
    for label_path in SOURCE_DIR.glob("*.txt"):
        if has_class_three(label_path):
            continue
        image_path = find_image(label_path)
        if image_path is None:
            missing_image_count += 1
            continue
        eligible_pairs.append((image_path, label_path))

    if len(eligible_pairs) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Beklenen {EXPECTED_IMAGE_COUNT} görsel-etiket çifti; "
            f"bulunan: {len(eligible_pairs)}."
        )

    destinations = [
        DESTINATION_DIR / "test",
        DESTINATION_DIR / "val",
        *(DESTINATION_DIR / "train" / name for name in PART_DIRECTORIES),
    ]
    for destination in destinations:
        for subdirectory in ("images", "labels"):
            output_directory = destination / subdirectory
            if output_directory.exists() and any(output_directory.iterdir()):
                raise FileExistsError(
                    f"Hedef klasör boş değil; üzerine yazılmayacak: {output_directory}"
                )

    random.shuffle(eligible_pairs)
    train_count = len(eligible_pairs) * 80 // 100
    test_count = (len(eligible_pairs) - train_count + 1) // 2
    val_count = len(eligible_pairs) - train_count - test_count

    train_pairs = eligible_pairs[:train_count]
    test_pairs = eligible_pairs[train_count : train_count + test_count]
    val_pairs = eligible_pairs[train_count + test_count :]

    copy_pairs(test_pairs, DESTINATION_DIR / "test")
    copy_pairs(val_pairs, DESTINATION_DIR / "val")

    chunk_size = len(train_pairs) // len(PART_DIRECTORIES)
    for index, directory_name in enumerate(PART_DIRECTORIES):
        start = index * chunk_size
        end = (
            len(train_pairs)
            if index == len(PART_DIRECTORIES) - 1
            else start + chunk_size
        )
        copy_pairs(
            train_pairs[start:end],
            DESTINATION_DIR / "train" / directory_name,
        )

    print(f"3. sınıfı olmayan ve görseli bulunan çift: {len(eligible_pairs)}")
    print(f"Görseli bulunmadığı için atlanan etiket: {missing_image_count}")
    print(f"Train: {len(train_pairs)}; test: {len(test_pairs)}; val: {len(val_pairs)}")
    print(
        "Train parçaları: "
        + ", ".join(
            f"{name}={len(train_pairs) // len(PART_DIRECTORIES)}"
            for name in PART_DIRECTORIES[:-1]
        )
        + f", {PART_DIRECTORIES[-1]}="
        + str(
            len(train_pairs)
            - (len(train_pairs) // len(PART_DIRECTORIES))
            * (len(PART_DIRECTORIES) - 1)
        )
    )


if __name__ == "__main__":
    main()