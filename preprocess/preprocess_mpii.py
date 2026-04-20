"""
MPII 人体姿态估计数据集预处理脚本。处理 .mat 标注文件，准备 MPII 的 GT 测试集格式。
"""
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


def write_json(dst: Path, obj: object, dry_run: bool) -> None:
    ensure_dir(dst.parent)
    if dry_run:
        return
    with dst.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def clamp_visible(v: float, x: float, y: float) -> int:
    if x is None or y is None:
        return 0
    if x < 0 or y < 0:
        return 0
    return 1 if v and v > 0 else 0


def convert_record(raw: dict) -> dict:
    joints_raw = raw.get("joint_self", [])
    joints = []
    joints_vis = []
    for j in joints_raw:
        if not isinstance(j, (list, tuple)) or len(j) < 3:
            joints.append([0.0, 0.0])
            joints_vis.append(0)
            continue
        x = float(j[0])
        y = float(j[1])
        v = float(j[2])
        joints.append([x, y])
        joints_vis.append(clamp_visible(v, x, y))
    return {
        "image": raw["img_paths"],
        "center": [float(raw["objpos"][0]), float(raw["objpos"][1])],
        "scale": float(raw["scale_provided"]),
        "joints": joints,
        "joints_vis": joints_vis,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path(r"e:\PoseBH\dataset\mpii_human_pose"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(r"e:\PoseBH\preprocess\mpii"),
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

    src_images_dir = src_root / "images"
    out_images_dir = out_root / "images"
    if mode == "link":
        if src_images_dir.exists():
            ok = try_create_dir_link(src_images_dir, out_images_dir, dry_run)
            if not ok and not out_images_dir.exists():
                ensure_dir(out_images_dir)
        else:
            ensure_dir(out_images_dir)
    else:
        remove_link_path(out_images_dir, dry_run)
        if mode == "copy":
            safe_remove_any_dir(out_images_dir, dry_run)
            if not dry_run and src_images_dir.exists():
                shutil.copytree(src_images_dir, out_images_dir, dirs_exist_ok=True)
            else:
                ensure_dir(out_images_dir)
        else:
            hardlink_or_copy_tree(src_images_dir, out_images_dir, dry_run)

    src_ann_path = src_root / "mpii_annotations.json"
    out_ann_dir = out_root / "annotations"
    ensure_dir(out_ann_dir)

    if src_ann_path.exists():
        if dry_run:
            train_out = out_ann_dir / "mpii_train.json"
            val_out = out_ann_dir / "mpii_val.json"
            test_out = out_ann_dir / "mpii_test.json"
            ensure_dir(out_ann_dir)
            for p in (train_out, val_out, test_out):
                if not p.exists():
                    write_json(p, [], dry_run)
            return

        with src_ann_path.open("r", encoding="utf-8") as f:
            raw_anno = json.load(f)

        train_anno = []
        val_anno = []
        for raw in raw_anno:
            if "img_paths" not in raw or "objpos" not in raw or "scale_provided" not in raw:
                continue
            record = convert_record(raw)
            if float(raw.get("isValidation", 0.0)) >= 0.5:
                val_anno.append(record)
            else:
                train_anno.append(record)

        write_json(out_ann_dir / "mpii_train.json", train_anno, dry_run)
        write_json(out_ann_dir / "mpii_val.json", val_anno, dry_run)
        test_anno = [{"image": x["image"], "center": x["center"], "scale": x["scale"]} for x in val_anno]
        write_json(out_ann_dir / "mpii_test.json", test_anno, dry_run)

    src_csv = src_root / "mpii_human_pose.csv"
    if src_csv.exists():
        copy_file(src_csv, out_root / "mpii_human_pose.csv", dry_run)


if __name__ == "__main__":
    main()
