# EGA: Euclidean Geodesic Alignment

We aim to establish a new paradigm that bridges the gap between upstream representation learning and downstream approximate nearest neighbor search. By introducing a lightweight manifold alignment layer, EGA smooths the highly twisted semantic space into a geometrically navigable Euclidean space.

## Quick Start
```bash
cd ~/hpdic # or any directory you prefer
git clone https://github.com/hpdic/EGA.git
cd ~/hpdic/EGA
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy tqdm
pip install git+https://github.com/openai/CLIP.git
pip install --upgrade pip
python ./scripts/01_extract_vit_features.py 

pip install faiss-cpu
python ./scripts/03_eval_ivf.py
```