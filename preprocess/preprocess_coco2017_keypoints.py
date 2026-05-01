import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def _ps_single_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if dry_run:
        return
    shutil.copy2(src, dst)


def try_create_junction(src_dir: Path, dst_dir: Path, dry_run: bool) -> bool:
    if dst_dir.exists():
        return dst_dir.is_dir()
    ensure_dir(dst_dir.parent)
    if dry_run:
        return True
    cmd = (
        "New-Item -ItemType Junction -Force -Path "
        + _ps_single_quote(str(dst_dir))
        + " -Target "
        + _ps_single_quote(str(src_dir))
        + " | Out-Null"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def try_create_symlink(src_path: Path, dst_path: Path, dry_run: bool) -> bool:
    if dst_path.exists() or dst_path.is_symlink():
        return True
    ensure_dir(dst_path.parent)
    if dry_run:
        return True
    try:
        os.symlink(src_path, dst_path, target_is_directory=src_path.is_dir())
        return True
    except Exception:
        return False


def try_create_dir_link(src_dir: Path, dst_dir: Path, dry_run: bool) -> bool:
    if os.name == "nt":
        return try_create_junction(src_dir, dst_dir, dry_run)
    return try_create_symlink(src_dir, dst_dir, dry_run)


def hardlink_or_copy_tree(src_dir: Path, dst_dir: Path, dry_run: bool) -> None:
    if not src_dir.exists():
        ensure_dir(dst_dir)
        return
    ensure_dir(dst_dir)
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        rel = root_path.relative_to(src_dir)
        out_root = dst_dir / rel
        ensure_dir(out_root)
        for d in dirs:
            ensure_dir(out_root / d)
        for name in files:
            src_file = root_path / name
            dst_file = out_root / name
            if dst_file.exists():
                continue
            if dry_run:
                continue
            try:
                os.link(src_file, dst_file)
            except Exception:
                shutil.copy2(src_file, dst_file)


def is_reparse_point(path: Path) -> bool:
    if not path.exists():
        return False
    if os.name != "nt":
        return path.is_symlink()
    try:
        st = os.stat(str(path), follow_symlinks=False)
        return bool(st.st_file_attributes & 0x0400)
    except Exception:
        return False


def remove_link_path(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if not is_reparse_point(path):
        return
    if dry_run:
        return
    if path.is_dir():
        os.rmdir(path)
        return
    path.unlink()


def safe_remove_any_dir(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        return
    if is_reparse_point(path):
        os.rmdir(path)
        return
    shutil.rmtree(path)


def safe_rename_dir(path: Path, suffix: str, dry_run: bool) -> Path:
    if not path.exists():
        return path
    if dry_run:
        return path
    if is_reparse_point(path):
        os.rmdir(path)
        return path
    new_path = path.with_name(path.name + suffix)
    i = 0
    while new_path.exists():
        i += 1
        new_path = path.with_name(path.name + f"{suffix}_{i}")
    path.rename(new_path)
    return new_path


def robocopy_copy_dir(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    cmd = [
        "robocopy",
        str(src_dir),
        str(dst_dir),
        "/E",
        "/R:1",
        "/W:1",
        "/MT:16",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NC",
        "/NS",
        "/NP",
    ]
    p = subprocess.run(cmd)
    if p.returncode > 7:
        raise RuntimeError(f"robocopy failed with exit code {p.returncode}")


def write_json(dst: Path, obj: object, dry_run: bool) -> None:
    ensure_dir(dst.parent)
    if dry_run:
        return
    with dst.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path(r"e:\PoseBH\dataset\coco2017-keypoints"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(r"e:\PoseBH\preprocess\coco"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["hardlink", "copy", "link"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_root: Path = args.src_root
    out_root: Path = args.out_root
    mode: str = args.mode
    dry_run: bool = args.dry_run

    src_ann_dir = src_root / "annotations"
    out_ann_dir = out_root / "annotations"

    ensure_dir(out_ann_dir)
    copy_file(
        src_ann_dir / "person_keypoints_train2017.json",
        out_ann_dir / "person_keypoints_train2017.json",
        dry_run,
    )
    copy_file(
        src_ann_dir / "person_keypoints_val2017.json",
        out_ann_dir / "person_keypoints_val2017.json",
        dry_run,
    )

    src_train_dir = src_root / "train2017"
    src_val_dir = src_root / "val2017"
    out_train_dir = out_root / "train2017"
    out_val_dir = out_root / "val2017"
    out_test_dir = out_root / "test2017"

    if mode == "link":
        if src_train_dir.exists():
            ok = try_create_dir_link(src_train_dir, out_train_dir, dry_run)
            if not ok and not out_train_dir.exists():
                ensure_dir(out_train_dir)
        else:
            ensure_dir(out_train_dir)

        if src_val_dir.exists():
            ok = try_create_dir_link(src_val_dir, out_val_dir, dry_run)
            if not ok and not out_val_dir.exists():
                ensure_dir(out_val_dir)
            ok = try_create_dir_link(src_val_dir, out_test_dir, dry_run)
            if not ok and not out_test_dir.exists():
                ensure_dir(out_test_dir)
        else:
            ensure_dir(out_val_dir)
            ensure_dir(out_test_dir)
    else:
        remove_link_path(out_train_dir, dry_run)
        remove_link_path(out_val_dir, dry_run)
        remove_link_path(out_test_dir, dry_run)

        if mode == "copy":
            safe_rename_dir(out_train_dir, "_old", dry_run)
            safe_rename_dir(out_val_dir, "_old", dry_run)
            safe_rename_dir(out_test_dir, "_old", dry_run)
            if dry_run:
                ensure_dir(out_train_dir)
                ensure_dir(out_val_dir)
                ensure_dir(out_test_dir)
            else:
                if src_train_dir.exists():
                    if os.name == "nt":
                        robocopy_copy_dir(src_train_dir, out_train_dir)
                    else:
                        shutil.copytree(src_train_dir, out_train_dir, dirs_exist_ok=True)
                else:
                    ensure_dir(out_train_dir)

                if src_val_dir.exists():
                    if os.name == "nt":
                        robocopy_copy_dir(src_val_dir, out_val_dir)
                        robocopy_copy_dir(src_val_dir, out_test_dir)
                    else:
                        shutil.copytree(src_val_dir, out_val_dir, dirs_exist_ok=True)
                        shutil.copytree(src_val_dir, out_test_dir, dirs_exist_ok=True)
                else:
                    ensure_dir(out_val_dir)
                    ensure_dir(out_test_dir)
        else:
            hardlink_or_copy_tree(src_train_dir, out_train_dir, dry_run)
            hardlink_or_copy_tree(src_val_dir, out_val_dir, dry_run)

        if mode != "copy":
            ok = try_create_dir_link(out_val_dir, out_test_dir, dry_run)
            if not ok and not out_test_dir.exists():
                hardlink_or_copy_tree(src_val_dir, out_test_dir, dry_run)

    test_ann_path = out_ann_dir / "person_keypoints_test2017.json"
    if not test_ann_path.exists():
        val_ann_path = out_ann_dir / "person_keypoints_val2017.json"
        if val_ann_path.exists():
            copy_file(val_ann_path, test_ann_path, dry_run)
        else:
            write_json(test_ann_path, {"images": [], "annotations": [], "categories": []}, dry_run)


if __name__ == "__main__":
    main()

