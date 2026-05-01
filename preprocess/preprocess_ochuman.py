import argparse
import shutil
import zipfile
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if dry_run:
        return
    shutil.copy2(src, dst)


def extract_zip(zip_path: Path, out_dir: Path, dry_run: bool) -> None:
    ensure_dir(out_dir)
    if dry_run:
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def flatten_images_dir(out_images_dir: Path, dry_run: bool) -> None:
    nested = out_images_dir / "images"
    if not nested.exists() or not nested.is_dir():
        return
    has_jpg_top = any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in out_images_dir.iterdir() if p.is_file())
    if has_jpg_top:
        return
    if dry_run:
        return
    for p in nested.iterdir():
        target = out_images_dir / p.name
        if target.exists():
            continue
        p.replace(target)
    try:
        nested.rmdir()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path(r"e:\PoseBH\dataset\OCHuman"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(r"e:\PoseBH\preprocess\ochuman"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_root: Path = args.src_root
    out_root: Path = args.out_root
    dry_run: bool = args.dry_run

    out_ann_dir = out_root / "annotations"
    ensure_dir(out_ann_dir)

    src_val = src_root / "ochuman_coco_format_val_range_0.00_1.00.json"
    src_test = src_root / "ochuman_coco_format_test_range_0.00_1.00.json"

    if src_val.exists():
        copy_file(src_val, out_ann_dir / "val.json", dry_run)
        copy_file(src_val, out_ann_dir / "train.json", dry_run)
    if src_test.exists():
        copy_file(src_test, out_ann_dir / "test.json", dry_run)

    images_zip = src_root / "images.zip"
    out_images_dir = out_root / "images"
    if images_zip.exists():
        ensure_dir(out_images_dir)
        has_any = any(
            p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for p in out_images_dir.rglob("*")
            if p.is_file()
        )
        if not has_any:
            extract_zip(images_zip, out_images_dir, dry_run)
        flatten_images_dir(out_images_dir, dry_run)
    else:
        ensure_dir(out_images_dir)

    src_raw = src_root / "ochuman.json"
    if src_raw.exists():
        copy_file(src_raw, out_root / "ochuman.json", dry_run)


if __name__ == "__main__":
    main()
