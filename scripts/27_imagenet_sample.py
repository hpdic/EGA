# scripts/sample_imagenet100.py
# 从 ImageNet-mini 随机采样 100 类（seed=42）

import os
import shutil
import random
from pathlib import Path

def sample_imagenet100(source_dir, target_dir, num_classes=100, seed=42):
    random.seed(seed)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有类别
    all_classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    print(f"Total classes in ImageNet-mini: {len(all_classes)}")
    
    # 随机采样 100 类
    selected_classes = random.sample(all_classes, num_classes)
    print(f"Selected {num_classes} classes")
    
    # 复制选中的类别
    for cls in selected_classes:
        src_cls_path = source_path / cls
        tgt_cls_path = target_path / cls
        
        if src_cls_path.exists():
            shutil.copytree(src_cls_path, tgt_cls_path)
            print(f"  Copied: {cls}")
    
    print(f"\nDone! Sampled {num_classes} classes to {target_dir}")

if __name__ == '__main__':
    source_dir = "/home/cc/hpdic/EGA/data/imagenet100/imagenet-mini/train"
    target_dir = "/home/cc/hpdic/EGA/data/imagenet100/imagenet100_train"
    
    sample_imagenet100(source_dir, target_dir, num_classes=100, seed=42)