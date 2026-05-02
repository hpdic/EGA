# EGA: Euclidean Geodesic Alignment

We aim to establish a new paradigm that bridges the gap between upstream representation learning and downstream approximate nearest neighbor search. By introducing a lightweight manifold alignment layer, EGA smooths the highly twisted semantic space into a geometrically navigable Euclidean space.

## Technical Reports
* [Preprint]({https://arxiv.org/abs/TBD})
* [Latest draft](./paper/EGA.pdf)

## Quick Start
```bash
cd ~/hpdic # or any directory you prefer
git clone https://github.com/hpdic/EGA.git
cd ~/hpdic/EGA
python3 -m venv venv
source venv/bin/activate
pip install numpy tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/openai/CLIP.git
pip install faiss-cpu
pip install scikit-learn matplotlib umap-learn
pip install --upgrade pip

touch ~/hpdic/EGA/models/__init__.py
export PYTHONPATH=$PYTHONPATH:$(pwd)

########################
# CIFAR-100 & CIFAR-10 #
########################

# CLIP ViT-B/32
python ./scripts/01_extract_features.py

# EGA (ours)
python ./scripts/02_train_ega.py
python ./scripts/03_eval_label_prediction.py
python ./scripts/04_eval_anns_recall.py
python ./scripts/05_visualize_precision.py
python ./scripts/06_visualize_distance.py
python ./scripts/07_visualize_recall.py
python ./scripts/08_visualize_projection.py

# ICon (ICLR-2025)
python ./scripts/ICon/00_train_icon.py
python ./scripts/ICon/01_extract_features_icon.py
python ./scripts/ICon/02_extract_features_icon_cifar10.py
python ./scripts/ICon/03_eval_recall_icon_cifar10.py

# SRL (CVPR-2025)
python ./scripts/SRL/00_train_srl.py
python ./scripts/SRL/01_extract_features_srl.py
python ./scripts/SRL/02_extract_features_srl_cifar10.py
python ./scripts/SRL/03_eval_recall_srl_cifar10.py

# Summary
python ./scripts/09_visualize_transfer.py
python ./scripts/10_ablation.py
python ./scripts/11_sensitivity.py
# There is an increasing list of scripts for more analyses, please check the `scripts` directory for the latest updates.

############
# Aircraft #
############

cd ~/hpdic/EGA
export PYTHONPATH=$PYTHONPATH:$(pwd)
python scripts/Aircraft/train_eval_aircraft.py
python scripts/Aircraft/plot_aircraft.py
python scripts/12_visualize_egaLoss.py

############
# Food-101 #
############

cd ~/hpdic/EGA
export PYTHONPATH=$PYTHONPATH:$(pwd)
python scripts/Food/train_eval_food.py
python scripts/Food/plot_food.py
```

## Contact
* Author: Dongfang Zhao, University of Washington, USA
* Email: dzhao@uw.edu
