# 项目名
CCL-CM: Cognitive-concept learning and conflict monitering integration

## Abstract
Explainable models based on concept learning can provide relatively intuitive explanations for decisions and have certain theoretical appeal. However, in practical applications, especially in classification tasks, they often face challenges in accuracy and generalization ability. Inspired by cognitive conflict theory, we propose a novel optimization paradigm that integrates cognitive concept learning and conflict detection behavior. By introducing the spatial representation mechanism from cognitive space theory, we optimize the parameter adjustment of concept learning models. This approach enhances the model's cognitive behavior and improves its concept learning capabilities by integrating multimodal-information. Additionally, conflict monitoring serves as a feedback mechanism, helping the model adjust its learning strategy when errors or conflicts are detected, improving the generalization performance of interpretable models. We have validated this method on four datasets: FashionMNIST, Cifar10, MNIST, and SVHN, achieving higher classification accuracy without sacrificing model interpretability and stability.

## Usage
#### Data Set
Download FashionMINST or MNIST or Cifar10 or SVHN and set them into direction of your "dataset_dir". You can also make your own dataset with the structure similar to Cifar10 and name it as Custom.

# MNIST 
python train_CCL_CM.py --dataset mnist --num-concepts 5 --epochs 20 --batch-size 64 --lr 2e-4

# CIFAR-10 
python train_CCL_CM.py --dataset cifar10 --num-concepts 5 --epochs 50 --batch-size 128 --lr 2e-4

# Fashion-MNIST
python train_CCL_CM.py --dataset FashionMNIST --num-concepts 5 --epochs 20

# SVHN
python train_CCL_CM.py --dataset SVHN --num-concepts 5 --epochs 20

A PyTorch implementation of an interpretable-by-design model that  
1. learns **semantic concepts** via an auto-encoder branch supervised by **CLIP** visual features,  
2. produces **instance-specific concept relevance** through a light-weight parametrizer,  
3. trains with a **Jacobian-matching regulariser (CM)** to guarantee concept faithfulness,  
4. and keeps reconstruction quality via an **SSIM sequence loss**.

---

## 🔍 What’s inside
| File | Purpose |
|------|---------|
| `CCL.py` | Core model (`SENN`, `ConceptAutoencoder`, `RelevanceParametrizer`, `Aggregator`) |
| `CM.py` | `parametriser_regulariser` – enforces gradient-equivalence between concepts and logits |
| `ssim_loss.py` | 1-D SSIM loss for **sequence-shaped** reconstruction (works on vectorised images) |
| `train_CCL_CM.py` | Full training / evaluation loop with TensorBoard logging |
| `requirements.txt` | One-command install (see below) |

---
## ⚙️ Setup
```bash
git clone https://github.com/asken-zhang/CCL-CM.git
cd SENN-CLIP
conda create -n senn python=3.9 -y && conda activate senn
pip install -r requirements.txt   # torch, torchvision, clip, tqdm, tensorboard
