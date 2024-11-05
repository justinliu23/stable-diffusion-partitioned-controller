# stable-diffusion-partitioned-controller

This repository implements a personalized Stable Diffusion model using DreamBooth fine-tuning, allowing for subject-driven image generation. Additionally, it introduces horizontal and vertical partitioned image controllers, which enable text-guided control over specific regions within an image for enhanced customization.

## Project Overview

This project combines DreamBooth-based personalization with advanced Stable Diffusion features, allowing users to generate images that reflect unique subjects while controlling specific image regions. By providing a few personal images, the model learns to generate images that incorporate these unique assets. The partitioned controllers allow for precise, text-driven customization of horizontal and vertical sections within generated images.

### Key Features

- **DreamBooth Fine-Tuning**: Leverages techniques from *"DreamBooth: Fine-Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"* to personalize Stable Diffusion. By providing a few images of a specific subject, users can fine-tune the model to recognize and recreate that subject in new contexts.
- **Horizontal and Vertical Partitioned Image Control**: The partitioned image controllers allow for text-prompt-based control over specific regions in generated images, either horizontally or vertically. This enables the application of different prompts to distinct areas, supporting more detailed and structured compositions.
- **Prior Preservation**: Includes a regularization technique to prevent overfitting during personalization, ensuring that the model can still generate generic objects alongside personalized subjects.
- **Memory Optimization**: Uses mixed-precision training (fp16), gradient checkpointing, and 8-bit optimizers to efficiently manage memory usage, making the process feasible on resource-limited devices.

## Core Components

- **DreamBoothDataset**: A custom dataset class that loads personalized images and prompts, and optionally regularization images, for fine-tuning.
- **PromptDataset**: Generates prompt-based data for consistent prompt input across samples.
- **collate_fn**: Prepares training batches by combining instance images and prompts, along with regularization images if prior preservation is enabled.
- **HorizontalPartitionedImageController and VerticalPartitionedImageController**: Classes implementing region-specific control in generated images, applying different prompts to distinct horizontal or vertical segments of the image.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- diffusers 0.30.3
- transformers 4.44.2
- accelerate 0.34.2
- xformers 0.0.28
- bitsandbytes (for memory-efficient 8-bit optimizers)
- torchvision, matplotlib, and other supporting libraries
