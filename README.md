# Fixing-the-Synthetic-Data-for-Representational-Learning

## 🧠 Overview

This project investigates the **limitations of synthetic data**—especially from diffusion models—in training deep learning models for computer vision. We aim to understand **why synthetic data underperforms** and propose techniques like **Diffusion Inversion** to close the realism gap between synthetic and real images.

---

## 🚧 Problem Statement

While synthetic data from GANs and diffusion models offers a scalable alternative to real datasets, it often leads to **lower performance** in downstream tasks due to:
- Missing fine-grained geometric and contextual details
- Misaligned vanishing points and shadows
- Averaging out rare or edge-case features

Our goal is to **enhance the representational quality of synthetic images** so that they become more useful for training high-performance vision models.

---

## 📚 Literature Insights

- **Geometric Discrepancies:** Generative models fail to preserve key projective geometry, such as vanishing points.
- **Average Representations:** Synthetic images often focus on “average” samples, missing out on edge cases.
- **Domain Gap:** Real and synthetic datasets differ in subtle but critical ways.
- **Diffusion Inversion:** A technique to condition image generation on real data distribution to improve realism and utility.

### References:
- [Shadows Don’t Lie...](https://projective-geometry.github.io/)
- [AI-Generated Images as Data Sources](https://arxiv.org/pdf/2310.01830)
- [Training on Thin Air](https://sites.google.com/view/diffusion-inversion)
- [Mind the Gap Between Synthetic and Real](https://arxiv.org/pdf/2405.03243)

---

## 🧪 Methodology

- We used **CIFAR-10** as the base dataset (10 image classes).
- Generated synthetic data using:
  - **Stable Diffusion**
  - **Diffusion Inversion** (with frozen model + optimized embeddings)
- Trained a **ResNet-18** classifier separately on:
  - Real data
  - Stable Diffusion data
  - Diffusion Inversion data
- Visualized model focus using **Grad-CAM**
- Evaluated geometric consistency via **projective geometry** analysis

---

## 📊 Results

| Training Data             | Accuracy (%) |
|---------------------------|--------------|
| Real                      | 81.80        |
| Diffusion Inversion       | 78.63        |
| Stable Diffusion          | 52.57        |
| Real + Diffusion Inversion| **86.14**    |
| Real + Stable Diffusion   | 83.94        |

### Observations:
- Diffusion Inversion improves realism and training utility over Stable Diffusion.
- Combining real and synthetic data results in **higher accuracy** than real alone.
- Attention maps (via Grad-CAM) show more localized focus in synthetic data.

---

## 📌 Key Takeaways

- Synthetic data must **respect projective geometry** to be truly effective.
- Techniques like **Diffusion Inversion** help bridge the realism gap.
- Carefully augmented synthetic data can **enhance model training**, especially in low-data or privacy-sensitive domains.

---

## 🔮 Future Applications

- Domain adaptation in low-data regimes (e.g., medical, autonomous driving)
- Training models in privacy-sensitive areas using synthetic-only data
- Using Grad-CAM feedback and geometry cues to guide better synthetic data generation

---

## 🖼️ Sample Visualizations

> Real vs Diffusion Inversion vs Stable Diffusion  
> Vanishing Point & Grad-CAM Heatmap Comparisons

*(Add images here if available)*

---

## 📂 Project Structure

├── data/ # Real & synthetic datasets
├── models/ # ResNet training scripts
├── analysis/ # Grad-CAM, geometry evaluation
├── results/ # Accuracy tables, visualizations
├── README.md # This file
└── .gitignore # Ignored files (check for .pt, .csv, etc.)

yaml
Copy
Edit

---

## 🧠 Want to Learn More?

Check out:
- [Diffusion Inversion](https://sites.google.com/view/diffusion-inversion)
- [Projective Geometry Visual Tools](https://projective-geometry.github.io/)
