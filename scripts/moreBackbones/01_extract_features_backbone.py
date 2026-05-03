# scripts/moreBackbones/01_extract_features_backbone.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CIFAR100
import argparse

def get_model_and_processor(backbone: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if backbone == "clip":
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess, 512, "clip"
    
    elif backbone.startswith("dinov2"):
        from transformers import AutoImageProcessor, AutoModel
        model_name = "facebook/dinov2-large" if "large" in backbone else "facebook/dinov2-base"
        dim = 1024 if "large" in backbone else 768
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        return model, processor, dim, "dinov2"
    
    elif backbone == "siglip":
        from transformers import AutoImageProcessor, AutoModel
        model_name = "google/siglip-so400m-patch14-384"
        dim = 1152
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        return model, processor, dim, "siglip"
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2-large",
                        choices=["clip", "dinov2-base", "dinov2-large", "siglip"])
    args = parser.parse_args()

    base_dir = os.path.expanduser("~/hpdic/EGA")
    embed_dir = os.path.join(base_dir, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)

    suffix = args.backbone.replace("-", "_")
    features_path = os.path.join(embed_dir, f"cifar100_{suffix}_features.npy")
    labels_path = os.path.join(embed_dir, "cifar100_labels.npy")

    if os.path.exists(features_path):
        print(f"✅ {args.backbone} features already exist: {features_path}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor, dim, model_type = get_model_and_processor(args.backbone)

    dataset = CIFAR100(root=os.path.join(base_dir, "data"), download=True, train=False)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for img, label in tqdm(dataset, desc=f"Extracting {args.backbone}"):
            if model_type == "clip":
                img_tensor = processor(img).unsqueeze(0).to(device)
                feature = model.encode_image(img_tensor)
            elif model_type == "siglip":
                inputs = processor(images=img, return_tensors="pt").to(device)
                # ✅ 关键修复：正确提取 SigLIP image features
                outputs = model.get_image_features(pixel_values=inputs.pixel_values)
                feature = outputs  # 通常是 tensor
                if not torch.is_tensor(feature):
                    # 兼容某些版本返回的 output 对象
                    feature = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
            else:  # dinov2
                inputs = processor(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
                feature = outputs.last_hidden_state[:, 0]

            all_features.append(feature.cpu().numpy())
            all_labels.append(label)

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels)

    np.save(features_path, features_array)
    np.save(labels_path, labels_array)
    print(f"✅ Saved {args.backbone} features (dim={dim}) → {features_path}")

if __name__ == "__main__":
    main()