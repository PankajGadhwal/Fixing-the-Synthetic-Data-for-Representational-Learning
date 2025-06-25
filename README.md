## Project Overview

This project investigates the limitations of synthetic data—especially data generated using diffusion models—for training deep learning models in computer vision. We focus on understanding why such data underperforms in downstream tasks and propose methods to improve its effectiveness. Specifically, we explore **Diffusion Inversion** as a technique to bridge the realism gap between synthetic and real-world images.

---

## Problem Statement

Deep learning models rely heavily on large, high-quality datasets. However, collecting real-world data is expensive, time-consuming, and often limited by privacy concerns. Synthetic data offers a scalable alternative but introduces a domain gap due to:

- Loss of geometric fidelity (e.g., misaligned vanishing points)
- Inconsistent shadows and textures
- Lack of contextual richness
- Over-representation of average cases, missing edge-case diversity

This project aims to enhance synthetic data quality to better support representation learning tasks.

---

## Literature Context

Key findings from related literature include:

- **Geometric Discrepancies:** Generative models often fail to respect projective geometry, leading to structural inconsistencies.
- **Representation Limitations:** Synthetic data lacks detailed context, impacting model generalization.
- **Improvement via Inversion:** Techniques like Diffusion Inversion help align synthetic samples with real image distributions.
- **Transfer Learning Gaps:** Domain shifts limit the transferability of models trained on synthetic datasets.

**Selected References:**
- [Shadows Don’t Lie and Lines Can’t Bend](https://projective-geometry.github.io/)
- [AI-Generated Images as Data Sources](https://arxiv.org/pdf/2310.01830)
- [Training on Thin Air](https://sites.google.com/view/diffusion-inversion)
- [Mind the Gap Between Synthetic and Real](https://arxiv.org/pdf/2405.03243)

---

## Methodology

- Dataset: CIFAR-10 (10 image classes)
- Generated synthetic datasets using:
  - Stable Diffusion
  - Diffusion Inversion
- Classifier: ResNet-18
- Training and evaluation performed separately on:
  - Real data
  - Stable Diffusion-generated data
  - Diffusion Inversion-generated data
- Grad-CAM and projective geometry tools used to visualize attention and structural integrity

---

## Results

| Training Set                | Test Accuracy (%) |
|----------------------------|-------------------|
| Real                       | 81.80             |
| Diffusion Inversion        | 78.63             |
| Stable Diffusion           | 52.57             |
| Real + Diffusion Inversion | 86.14             |
| Real + Stable Diffusion    | 83.94             |

### Additional Observations

- Grad-CAM revealed that models trained on fake images focused more narrowly, indicating a lack of diverse feature learning.
- Diffusion Inversion produced images that led to significantly better performance than Stable Diffusion.
- Combining real and synthetic data improved accuracy beyond using real data alone.

---

## Key Takeaways

- Projective geometry and visual attention analysis can reveal deep structural flaws in synthetic datasets.
- Diffusion Inversion is an effective approach for improving the realism and representational utility of synthetic images.
- Synthetic data, when properly aligned with real data distributions, can act as a valuable augmentation source in limited data or privacy-sensitive environments.

---

## Applications and Future Work

- Enhancing performance in domains with limited real data (e.g., medical imaging, autonomous driving)
- Privacy-preserving training using only synthetic data
- Integrating Grad-CAM feedback and geometric constraints into future generation pipelines

---
