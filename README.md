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

# CIFAR-100
python ./scripts/01_extract_features.py
python ./scripts/02_train_ega.py
python ./scripts/03_eval_label_prediction.py
python ./scripts/04_eval_anns_recall.py
python ./scripts/05_visualize_distance.py
python ./scripts/06_visualize_recall.py

## Contact
* Author: Dongfang Zhao, University of Washington, USA
* Email: dzhao@uw.edu
