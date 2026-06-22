# lung-segmentation-mobilenetv2-unet3plus-se

Lightweight Edge-AI framework for lung segmentation in CXRs. Combines MobileNetV2, UNet3+, and Squeeze-and-Excitation blocks with full-scale deep supervision. Achieves elite boundary precision (Dice >98.6%) with only 2.55M parameters. Validated across multi-center JSRT, Shenzhen, and Montgomery datasets. Includes Grad-CAM explainability.

## 🌟 Project Overview
This repository hosts the official implementation of a lightweight, highly optimized deep learning framework tailored for automated lung field segmentation in digital Chest X-Rays (CXRs).

The primary objective of this project is to break the conventional trade-off between clinical segmentation fidelity and computational complexity. By engineering an efficient hybrid architecture, this framework delivers state-of-the-art boundary precision while remaining perfectly viable for memory-constrained environments, Edge AI deployment, and point-of-care clinical devices.

## 🏗️ Architectural Breakdown
The framework's structure defines the synergy of three cutting-edge computer vision components:
* **MobileNetV2 (The Encoder / Backbone):** Serves as the primary feature extraction engine. Instead of relying on heavy, resource-intensive networks (such as VGG or ResNet), it utilizes Depthwise Separable Convolutions and inverted residual blocks. This drastically minimizes the parameter footprint and floating-point operations (FLOPs) without sacrificing representation capability.
* **UNet3+ (The Decoder):** Moving beyond the plain skip connections of standard UNet and the dense nested skips of UNet++, the UNet3+ topology introduces Full-scale Skip Connections. It seamlessly aggregates low-level fine-grained edge details with high-level global semantic positions across multiple scales, capturing rich spatial layouts.
* **Squeeze-and-Excitation (Attention Mechanisms):** Embedded directly within the feature recalibration pipelines, these attention blocks dynamically compute channel-wise relationships. The network explicitly learns to emphasize highly relevant anatomical boundaries (e.g., parenchymal lung borders) while actively suppressing background noise, artifacts, and uninformative structures.

## 🚀 Key Features & Functional Modules
* **Full-Scale Deep Supervision:** The decoding pathway integrates intermediate auxiliary loss branches at every scale layer. This forces early convergence, prevents vanishing gradients, and ensures mathematically rigorous boundary alignment.
* **Multi-Center Cross-Dataset Benchmarking:** Built to assess clinical generalizability and resilience against domain shifts, the codebase includes evaluation pipelines across major global benchmarks: Montgomery County (MC), Shenzhen (SZ), and JSRT.
* **Advanced Geometric & Boundary Evaluation:** In addition to standard overlap metrics (Dice, Jaccard/IoU, Accuracy), the framework natively computes distance-based boundary errors, specifically the 95% Hausdorff Distance (HD95) and the Average Symmetric Surface Distance (ASSD).
* **Visual Explainability (Grad-CAM):** Provides clinical transparency by demonstrating exactly where the model focuses its activation triggers during inference.

## 📦 Installation

Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/lung-segmentation-mobilenetv2-unet3plus-se.git](https://github.com/your-username/lung-segmentation-mobilenetv2-unet3plus-se.git)
cd lung-segmentation-mobilenetv2-unet3plus-se
pip install -r requirements.txt
