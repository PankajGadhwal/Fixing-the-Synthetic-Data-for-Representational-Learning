## Overview

This project studies and explores the limitations of synthetic data particularly that generated through diffusion models in training deep learning models for computer vision. It focuses on understanding the realism gap between synthetic and real images and proposes methods to improve the utility of synthetic data for representation learning. In particular, it explores Diffusion Inversion as a strategy to better align synthetic data distributions with real-world image characteristics.


## Problem Statement

Modern vision models require large volumes of high-quality labelled data. However, real-world data is often expensive to collect, time-consuming to annotate, and restricted by privacy concerns. Synthetic datasets generated through models such as GANs or diffusion-based generators offer scalable alternatives—but they frequently lead to performance degradation due to:

- Lack of geometric fidelity (e.g., misaligned vanishing points)
- Inconsistent lighting, shadows, and structural cues
- Lack of rare case representation
- Absence of contextual richness

This project aims to enhance the quality of synthetic datasets for downstream learning tasks, reducing the domain gap that limits their effectiveness.



## Literature Review

- **Geometric Discrepancies:** Generative models often fail to preserve fundamental structures like projective geometry and vanishing points.
- **Contextual Gaps:** Synthetic data tends to lack nuanced visual and spatial context.
- **Diffusion Inversion:** Conditioning synthetic image generation on real distributions can improve downstream classification performance.
- **Transfer Limitations:** Domain shifts between real and synthetic data present challenges in generalization and robustness.



## Methodology

- Dataset: CIFAR-10 (10 image classes)
- Synthetic data generation using:
  - Stable Diffusion
  - Diffusion Inversion (frozen diffusion model with optimized embeddings)
- Classifier: ResNet-18
- Training configurations:
  - Trained separately on real, Stable Diffusion, and Diffusion Inversion datasets
  - Combined real + synthetic datasets used to test performance gains
- Grad-CAM used to analyze spatial attention
- Projective geometry techniques applied to evaluate structural consistency



## Diffusion Inversion

- Adapts a frozen, pretrained diffusion model using a small set of trainable embedding vectors.
- Embeddings are injected into the U-Net at each timestep during the reverse diffusion process.
- Only the embeddings are optimised using gradient descent; model weights remain unchanged.
  ![image](https://github.com/user-attachments/assets/40205a86-62fd-4bfa-acda-aaa7510227d3)
  ![Screenshot 2025-07-08 094441](https://github.com/user-attachments/assets/22c9eb62-8b02-4401-aa1f-2d47f0103a84)



## Results

| Training Set                | Test Accuracy (%) |
|----------------------------|-------------------|
| Real                       | 81.80             |
| Diffusion Inversion        | 78.63             |
| Stable Diffusion           | 52.57             |
| Real + Diffusion Inversion | 86.14             |
| Real + Stable Diffusion    | 83.94             |



- **Diffusion Inversion outperformed Stable Diffusion** in downstream classification accuracy, but still fell short of real data.
- Combining real and synthetic data improved classification performance, validating the role of well-crafted synthetic augmentation.
- **Grad-CAM heatmaps** showed that synthetic-trained models had more localized attention, while real data encouraged more spatially distributed features.
- The **focus score and intensity** of real and inverted images were similar, but attention spread was narrower in synthetic data, indicating a reliance on limited features.



## Applications and Future Work

- Use in low-data or privacy-restricted domains such as healthcare or autonomous systems
- Improved synthetic generation guided by visual attention and projective geometry
- Fine-tuning later layers of models trained on synthetic data using small amounts of real data
- Broader application of Diffusion Inversion to other datasets and model architectures



## Contributors
  
- Pankaj
- Aayush Prakash Parmar
- Vatsal Trivedi  
- Astitva Aryan


## References

1. Shadows Don’t Lie and Lines Can’t Bend – [projective-geometry.github.io](https://projective-geometry.github.io/)
2. AI-Generated Images as Data Sources – [arXiv:2310.01830](https://arxiv.org/pdf/2310.01830)
3. Training on Thin Air – [Diffusion Inversion Site](https://sites.google.com/view/diffusion-inversion)
4. Mind the Gap Between Synthetic and Real – [arXiv:2405.03243](https://arxiv.org/pdf/2405.03243)
