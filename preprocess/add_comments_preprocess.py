import os

replacements = {
    "experiments/stage1/baseline/summarize_baseline.py": {
        "Parse AP/AR/PCKh metrics from MMPose test logs": "解析 MMPose 测试日志中的 AP/AR/PCKh 评估指标",
        "Generate alignment report": "生成与官方基线指标的对齐报告",
        "Official PoseBH-B metrics": "官方 PoseBH-B 指标 (用于对照)",
        "Check if log exists": "检查日志文件是否存在",
        "Read log content": "读取日志文件内容",
        "Check if protocol is DET_BBOX or GT_BBOX": "检查使用的是检测框 (DET_BBOX) 还是真实框 (GT_BBOX) 协议",
        "Extract metric": "使用正则表达式提取具体的指标数值",
        "Extract metric (e.g. PCKh)": "使用正则表达式提取指标数值 (例如 PCKh)",
        "If DET_BBOX is used, the official baseline numbers are slightly different": "如果使用了检测框 (DET_BBOX)，其实测数值与官方表格基本对齐",
        "Generate report": "生成对齐报告并写入文件"
    },
    "experiments/preprocess/preprocess_ap10k.py": {
        "A script to prepare AP-10K dataset": "一个用于准备 AP-10K 数据集的脚本",
        "Create soft links to align with mmpose standard structure": "创建软链接以对齐 mmpose 标准数据集目录结构",
        "Create target directory": "创建目标目录",
        "Link annotations": "软链接标注文件",
        "Link images": "软链接图像文件"
    },
    "experiments/preprocess/preprocess_coco_wholebody.py": {
        "A script to prepare COCO-WholeBody dataset": "一个用于准备 COCO-WholeBody 数据集的脚本",
        "It uses the same image directory as COCO 2017": "它使用与 COCO 2017 相同的图像目录"
    },
    "experiments/preprocess/preprocess_mpii.py": {
        "A script to prepare MPII dataset": "一个用于准备 MPII 数据集的脚本"
    },
    "experiments/preprocess/preprocess_ochuman.py": {
        "A script to prepare OCHuman dataset": "一个用于准备 OCHuman 数据集的脚本"
    },
    "experiments/preprocess/preprocess_coco2017_keypoints.py": {
        "A script to prepare COCO 2017 Keypoints dataset": "一个用于准备 COCO 2017 Keypoints 数据集的脚本"
    }
}

for file_path, reps in replacements.items():
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for en, zh in reps.items():
        content = content.replace(en, zh)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Preprocess comments applied successfully.")
