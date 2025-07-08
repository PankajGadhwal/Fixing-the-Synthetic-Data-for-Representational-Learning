## Overview

This project studies and explores the limitations of synthetic data particularly that generated through diffusion models in training deep learning models for computer vision. It focuses on understanding the realism gap between synthetic and real images and proposes methods to improve the utility of synthetic data for representation learning. In particular, it explores Diffusion Inversion as a strategy to better align synthetic data distributions with real-world image characteristics.


## Problem Statement

Modern vision models require large volumes of high-quality labelled data. However, real-world data is often expensive to collect, time-consuming to annotate, and restricted by privacy concerns. Synthetic datasets generated through models such as GANs or diffusion-based generators offer scalable alternatives but they frequently lead to performance degradation due to:

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

- Dataset: CIFAR-10/CIFAKE
- Synthetic data generation using Stable Diffusion and Diffusion Inversion 
- Classifier: ResNet-18
- Training configurations:
  - Trained separately on real, Stable Diffusion, and Diffusion Inversion datasets
  - Combined real + synthetic datasets used to test performance gains
- Grad-CAM used to analyze spatial attention
- Projective geometry techniques applied to evaluate structural consistency



## Diffusion Inversion

- Adapts a frozen, pre-trained diffusion model using a small set of trainable embedding vectors.
- Embeddings are injected into the U-Net at each timestep during the reverse diffusion process.
- Only the embeddings are optimised using gradient descent; model weights remain unchanged.
  ![image](https://github.com/user-attachments/assets/40205a86-62fd-4bfa-acda-aaa7510227d3)
- Minimises denoising loss between predicted and actual noise to align generations with real data.
- Enables the model to generate more realistic images while preserving its original architecture.
  


## Visualising the Realism Gap

![image](https://github.com/user-attachments/assets/5168b187-2eef-40e5-9b23-5df4df555c6e)



## Analysing Projective Geometry

 ![image](https://github.com/user-attachments/assets/381929ab-d26f-4c84-a8a8-69379efd5810)
- Vanishing points are key elements in projective geometry, where parallel lines in 3D space appear to converge in a 2D image, providing information about the spatial structure, depth, and orientation of scenes.
- Grad-CAM is a technique used to visualise which areas of an input image highlight important regions for a model's prediction, by generating heat maps. Warmer colours (red/yellow) in the heat maps indicate a      stronger influence.
- Focus Score and Intensity are almost identical, suggesting that the model is equally confident for both image types. Attention Spread is slightly lower for fake images, indicating that attention is slightly      more concentrated in certain regions for fake images. This suggests that fake images rely more on localised features.

| Dataset              | Focus Score | Attention Spread | Intensity |
|----------------------|-------------|------------------|-----------|
| Real Images          | 0.439       | 0.4177           | 0.439     |
| Fake Inverted Images | 0.4361      | 0.4068           | 0.4361    |


## Results

| Training Set                | Test Accuracy (%) |
|----------------------------|-------------------|
| Real                       | 81.80             |
| Diffusion Inversion        | 78.63             |
| Stable Diffusion           | 52.57             |
| Real + Diffusion Inversion | 86.14             |
| Real + Stable Diffusion    | 83.94             |



- Diffusion Inversion performed better than Stable Diffusion but worse than the real dataset, indicating that while it generates more realistic samples than Stable Diffusion, it still doesn't match the quality     of real data.
- The lower performance of Stable Diffusion is likely due to its inability to accurately capture the true data distribution, leading to less useful synthetic samples for
  training.
- Combining real data with synthetic data (data augmentation) led to higher accuracies, showing that well-crafted synthetic data can enhance model performance when used alongside real examples.



## Applications and Future Work

- Fine-tune only the later layers of synthetic-trend models to close the accuracy gap with minimal real data.
- Use projective geometry and Grad-CAM feedback to guide better synthetic generation.
- Apply refined synthetic data for domain adaptation in low-data areas like medical or autonomous driving.
- Train in privacy-sensitive domains using synthetic data to avoid exposing real personal information.



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
