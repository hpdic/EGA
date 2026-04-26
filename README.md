# EGA: Euclidean Geodesic Alignment

We aim to establish a new paradigm that bridges the gap between upstream representation learning and downstream approximate nearest neighbor search. By introducing a lightweight manifold alignment layer, EGA smooths the highly twisted semantic space into a geometrically navigable Euclidean space.

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
pip install --upgrade pip

touch ~/hpdic/EGA/models/__init__.py
export PYTHONPATH=$PYTHONPATH:$(pwd)

python ./scripts/01_extract_vit_features.py 
python ./scripts/02_train_ega.py
python ./scripts/03_eval_ivf.py

python ./scripts/04_extract_cifar100_features.py
python ./scripts/02_train_ega_cifar100.py
python ./scripts/05_eval_generalization.py
```