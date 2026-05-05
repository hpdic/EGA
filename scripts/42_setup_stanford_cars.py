#!/usr/bin/env python3
"""一键下载并组织 Stanford Cars 数据集，使其能被 torchvision 读取。"""
import os, shutil, tarfile, kagglehub

def setup(target_root="./data"):
    target = os.path.join(target_root, "stanford_cars")
    if os.path.exists(target):
        print(f"✅ {target} already exists. Skipping setup.")
        return

    print("📥 Downloading Stanford Cars dataset via KaggleHub...")
    # 下载 Kaggle 数据集（返回缓存路径）
    kaggle_path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
    print(f"   Cached at: {kaggle_path}")

    # 下载 devkit（来自 PyTorch Vision GitHub）
    devkit_url = "https://github.com/pytorch/vision/files/11644847/car_devkit.tgz"
    devkit_tgz = os.path.join("/tmp", "car_devkit.tgz")
    os.system(f"wget -q -O {devkit_tgz} {devkit_url}")
    if not os.path.exists(devkit_tgz):
        raise RuntimeError("Failed to download devkit")

    # 解压 devkit
    with tarfile.open(devkit_tgz, "r:gz") as f:
        f.extractall("/tmp")  # 解压到 /tmp/devkit/

    # 开始构建目标目录
    os.makedirs(target, exist_ok=True)

    # 1. 拷贝图像文件夹（cars_train, cars_test）
    for split in ["cars_train", "cars_test"]:
        src = os.path.join(kaggle_path, split)
        dst = os.path.join(target, split)
        if not os.path.exists(src):
            # 有些 Kaggle 版本子文件夹名可能不同，尝试查找
            alt = os.path.join(kaggle_path, "car_data", split)
            if os.path.exists(alt):
                src = alt
        shutil.copytree(src, dst)
        print(f"   Copied {split} images")

    # 2. 拷贝 devkit
    shutil.copytree("/tmp/devkit", os.path.join(target, "devkit"))

    # 3. 拷贝 cars_test_annos_withlabels.mat（从 Kaggle 下载中找）
    mat_src = None
    for root, dirs, files in os.walk(kaggle_path):
        if "cars_test_annos_withlabels.mat" in files:
            mat_src = os.path.join(root, "cars_test_annos_withlabels.mat")
            break
    if mat_src:
        shutil.copy(mat_src, target)
        print("   Copied cars_test_annos_withlabels.mat")
    else:
        # 若仍找不到，尝试从 devkit 里的 cars_test_annos.mat 转换？先报错
        raise RuntimeError("cars_test_annos_withlabels.mat not found in Kaggle dataset")

    # 清理临时文件
    os.remove(devkit_tgz)
    shutil.rmtree("/tmp/devkit", ignore_errors=True)
    print(f"✅ Dataset ready at {target}")

if __name__ == "__main__":
    setup()