from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "orijinal_veriler_2026"
TARGET_DIR = PROJECT_ROOT / "Klasor_1(Johsongrassli)"
CLASSES_FILE = SOURCE_DIR / "classes.txt"
CLASS_NAME = "Johnsongrass"
EXPECTED_CLASS_INDEX = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RANDOM_SEED = 42


def get_class_index() -> int:
    """Return Johnsongrass' class ID after checking the classes file."""
    if not CLASSES_FILE.is_file():
        raise FileNotFoundError(f"classes.txt bulunamadi: {CLASSES_FILE}")

    classes = [line.strip() for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()]
    if CLASS_NAME not in classes:
        raise ValueError(f"{CLASS_NAME} sinifi classes.txt dosyasinda yok.")

    class_index = classes.index(CLASS_NAME)
    if class_index != EXPECTED_CLASS_INDEX:
        raise ValueError(
            f"{CLASS_NAME} sinifinin indexi {class_index}; "
            f"beklenen index {EXPECTED_CLASS_INDEX}."
        )
    return class_index


def find_image_labels(class_index: int) -> list[tuple[Path, Path]]:
    """Find image/label pairs whose labels contain the requested class."""
    images_by_stem: dict[str, list[Path]] = {}
    for image_path in SOURCE_DIR.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_stem.setdefault(image_path.stem, []).append(image_path)

    pairs = []
    for label_path in SOURCE_DIR.glob("*.txt"):
        if label_path.name.lower() == CLASSES_FILE.name.lower():
            continue

        contains_class = False
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            try:
                label_class_index = int(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"Gecersiz sinif ID: {label_path.name}, satir {line_number}"
                ) from error
            if label_class_index == class_index:
                contains_class = True
                break

        if not contains_class:
            continue

        matching_images = images_by_stem.get(label_path.stem, [])
        if len(matching_images) != 1:
            raise ValueError(
                f"{label_path.name} icin tam bir gorsel bulunamadi "
                f"(bulunan gorsel sayisi: {len(matching_images)})."
            )
        pairs.append((matching_images[0], label_path))

    return sorted(pairs, key=lambda pair: pair[1].name.lower())


def split_pairs(
    pairs: list[tuple[Path, Path]],
) -> dict[str, list[tuple[Path, Path]]]:
    """Shuffle pairs deterministically and split them into 80/10/10."""
    shuffled_pairs = pairs.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled_pairs)

    train_count = int(len(shuffled_pairs) * 0.8)
    test_count = int(len(shuffled_pairs) * 0.1)
    return {
        "train": shuffled_pairs[:train_count],
        "test": shuffled_pairs[train_count : train_count + test_count],
        "val": shuffled_pairs[train_count + test_count :],
    }


def copy_splits(splits: dict[str, list[tuple[Path, Path]]]) -> None:
    """Copy each image and label into its split's images and labels folders."""
    for split_name, pairs in splits.items():
        images_dir = TARGET_DIR / split_name / "images"
        labels_dir = TARGET_DIR / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in pairs:
            shutil.copy2(image_path, images_dir / image_path.name)
            shutil.copy2(label_path, labels_dir / label_path.name)

        print(f"{split_name}: {len(pairs)} gorsel ve etiket kopyalandi.")


def main() -> None:
    class_index = get_class_index()
    pairs = find_image_labels(class_index)
    splits = split_pairs(pairs)

    print(f"Johnsongrass iceren gorsel/etiket cifti: {len(pairs)}")
    copy_splits(splits)
    print(f"Tamamlandi. Dosyalar: {TARGET_DIR}")


if __name__ == "__main__":
    main()