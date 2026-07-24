import os
import torch
import clip
import numpy as np
from datasets import load_dataset
from torchvision import datasets
from tqdm import tqdm

def extract_from_hf(dataset_name, hf_name, hf_split, model, preprocess, device, embed_dir, image_key='img', label_key='fine_label', batch_size=128):
    feat_path = os.path.join(embed_dir, f'{dataset_name}_features.npy')
    label_path = os.path.join(embed_dir, f'{dataset_name}_labels.npy')

    if os.path.exists(feat_path) and os.path.exists(label_path):
        print(f'[{dataset_name}] Features already exist at {feat_path}. Skipping.')
        return

    print(f'[{dataset_name}] Loading HF dataset {hf_name} split {hf_split}...')
    ds = load_dataset(hf_name, split=hf_split)
    print(f'[{dataset_name}] Extracting features for {len(ds)} samples...')

    all_feats = []
    all_labels = []

    for i in tqdm(range(0, len(ds), batch_size), desc=f'Encoding {dataset_name}'):
        batch = ds[i:i+batch_size]
        imgs = [img.convert('RGB') for img in batch[image_key]]
        lbls = batch[label_key]

        tensors = torch.stack([preprocess(img) for img in imgs]).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensors)

        all_feats.append(feats.cpu().numpy())
        all_labels.append(np.array(lbls))

    features_array = np.concatenate(all_feats, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    os.makedirs(embed_dir, exist_ok=True)
    np.save(feat_path, features_array)
    np.save(label_path, labels_array)
    print(f'[{dataset_name}] Saved features shape {features_array.shape} to {feat_path}')

def extract_from_torchvision(dataset_name, dataset, model, preprocess, device, embed_dir, batch_size=128):
    feat_path = os.path.join(embed_dir, f'{dataset_name}_features.npy')
    label_path = os.path.join(embed_dir, f'{dataset_name}_labels.npy')

    if os.path.exists(feat_path) and os.path.exists(label_path):
        print(f'[{dataset_name}] Features already exist at {feat_path}. Skipping.')
        return

    print(f'[{dataset_name}] Extracting features for {len(dataset)} samples...')
    all_feats = []
    all_labels = []

    for i in tqdm(range(0, len(dataset), batch_size), desc=f'Encoding {dataset_name}'):
        batch_imgs = []
        batch_lbls = []
        for j in range(i, min(i + batch_size, len(dataset))):
            img, lbl = dataset[j]
            batch_imgs.append(preprocess(img))
            batch_lbls.append(lbl)

        tensors = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensors)

        all_feats.append(feats.cpu().numpy())
        all_labels.append(np.array(batch_lbls))

    features_array = np.concatenate(all_feats, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    os.makedirs(embed_dir, exist_ok=True)
    np.save(feat_path, features_array)
    np.save(label_path, labels_array)
    print(f'[{dataset_name}] Saved features shape {features_array.shape} to {feat_path}')

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    data_dir = os.path.join(base_dir, 'data')
    embed_dir = os.path.join(base_dir, 'embeddings')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading CLIP ViT-B/32 on {device}...')
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    # 1. CIFAR-100 (test split = 10,000 samples)
    extract_from_hf('cifar100_vit_b32', 'cifar100', 'test', model, preprocess, device, embed_dir, image_key='img', label_key='fine_label')

    # 2. CIFAR-10 (test split = 10,000 samples)
    extract_from_hf('cifar10_vit_b32', 'cifar10', 'test', model, preprocess, device, embed_dir, image_key='img', label_key='label')

    # 3. FGVC Aircraft (test split = 3,333 samples)
    air_dataset = datasets.FGVCAircraft(root=data_dir, split='test', download=False)
    extract_from_torchvision('aircraft_test_vit_b32', air_dataset, model, preprocess, device, embed_dir)

    # 4. Food-101 (validation split = 25,250 samples)
    extract_from_hf('food101', 'food101', 'validation', model, preprocess, device, embed_dir, image_key='image', label_key='label')

    print('All required dataset features are ready!')

if __name__ == '__main__':
    main()
