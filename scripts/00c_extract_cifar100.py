import os
import torch
import clip
import numpy as np
from datasets import load_dataset
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

def main():
    base_dir = os.path.expanduser('~/hpdic/EGA')
    embed_dir = os.path.join(base_dir, 'embeddings')
    os.makedirs(embed_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading CLIP ViT-B/32 on {device}...')
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    # CIFAR-100 (test split = 10,000 samples) -- used as TRAIN set for the CIFAR-10 OOD protocol
    extract_from_hf('cifar100_vit_b32', 'uoft-cs/cifar100', 'test', model, preprocess, device, embed_dir, image_key='img', label_key='fine_label')

    print('CIFAR-100 features ready!')

if __name__ == '__main__':
    main()
