"""
AP-10K 动物姿态估计数据集预处理脚本。将标注文件转换为通用的评估格式。
"""
import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path(r"e:\PoseBH\dataset\ap-10k"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(r"e:\PoseBH\preprocess\ap10k"),
    )
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
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
    split: int = args.split
    mode: str = args.mode
    dry_run: bool = args.dry_run

    src_ann_dir = src_root / "annotations"
    out_ann_dir = out_root / "annotations"
    ensure_dir(out_ann_dir)

    for split_i in (1, 2, 3):
        for kind in ("train", "val", "test"):
            name = f"ap10k-{kind}-split{split_i}.json"
            copy_file(src_ann_dir / name, out_ann_dir / name, dry_run)

    src_data_dir = src_root / "data"
    out_data_dir = out_root / "data"
    if mode == "link":
        if src_data_dir.exists():
            ok = try_create_dir_link(src_data_dir, out_data_dir, dry_run)
            if not ok and not out_data_dir.exists():
                ensure_dir(out_data_dir)
        else:
            ensure_dir(out_data_dir)
    else:
        remove_link_path(out_data_dir, dry_run)
        if mode == "copy":
            safe_remove_any_dir(out_data_dir, dry_run)
            if not dry_run and src_data_dir.exists():
                shutil.copytree(src_data_dir, out_data_dir, dirs_exist_ok=True)
            else:
                ensure_dir(out_data_dir)
        else:
            hardlink_or_copy_tree(src_data_dir, out_data_dir, dry_run)

    copy_file(
        out_ann_dir / f"ap10k-train-split{split}.json",
        out_ann_dir / "train.json",
        dry_run,
    )
    copy_file(
        out_ann_dir / f"ap10k-val-split{split}.json",
        out_ann_dir / "val.json",
        dry_run,
    )
    copy_file(
        out_ann_dir / f"ap10k-test-split{split}.json",
        out_ann_dir / "test.json",
        dry_run,
    )

    paths_file = out_root / "paths.txt"
    if not paths_file.exists():
        ensure_dir(paths_file.parent)
        if not dry_run:
            paths_file.write_text(
                os.linesep.join(
                    [
                        f"img_prefix={out_data_dir}",
                        f"ann_train={out_ann_dir / 'train.json'}",
                        f"ann_val={out_ann_dir / 'val.json'}",
                        f"ann_test={out_ann_dir / 'test.json'}",
                    ]
                )
                + os.linesep,
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
