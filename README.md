# GCP FSDP Training Framework

A comprehensive framework for deploying and managing Fully Sharded Data Parallel (FSDP) distributed training workloads on Google Cloud Platform using GKE (Google Kubernetes Engine).

## Project Overview

This repository provides a complete solution for setting up and optimizing distributed deep learning training jobs on GCP, with a focus on PyTorch's FSDP implementation with ZeRO-3 optimizations. The framework is especially designed for efficient training on Tesla T4 GPU clusters.

### Key Features

- **Infrastructure as Code**: Complete Terraform configurations for GCP resource provisioning
- **FSDP Optimization**: Ready-to-use implementations for PyTorch's FSDP with ZeRO-3 optimizations
- **GKE Integration**: Kubernetes configurations optimized for distributed training workloads
- **Performance Tuning**: Best practices for GPU utilization, memory management, and communication optimization
- **Monitoring Solutions**: Built-in performance profiling and observability tools

## Repository Structure

This repository is organized into two main components:

### 1. [GCP Infrastructure as Code (IaaC)](./gcp_iaac/)

The `gcp_iaac` directory contains Terraform configurations and instructions for setting up your GCP infrastructure. This includes:

- GKE cluster provisioning with optimized T4 GPU node pools
- Networking setup for efficient node communication
- Storage configuration for training data and checkpoints
- IAM and security best practices

For setup instructions, see the [GCP IaaC README](./gcp_iaac/README.md).

### 2. [FSDP Implementation](./fsdp/)

The `fsdp` directory contains the distributed training framework implementing PyTorch's FSDP (Fully Sharded Data Parallel) with ZeRO-3 optimization. This includes:

- Reference implementations for various model architectures
- Optimized DataLoader configurations for T4 GPUs
- Memory management techniques (activation checkpointing, CPU offloading)
- Communication optimizations for distributed training
- Kubernetes deployment manifests for GKE

For implementation details and usage, see the [FSDP README](./fsdp/README.md).

## Hardware Specifications

This framework is optimized for the following hardware configuration:

- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **Node Type**: n1-standard-32 (32 vCPUs, 120GB RAM)
- **Storage**: 300GB per node
- **Cluster Size**: Configurable, with reference setup for 2+ nodes

## Getting Started

1. Clone this repository
   ```bash
   git clone https://github.com/faizananwar532/fsdp_training_on_gke_with_t4_gpus
   cd fsdp_training_on_gke_with_t4_gpus
   ```

2. Set up your GCP infrastructure following the instructions in the [GCP IaaC README](./gcp_iaac/README.md)

3. Deploy your distributed training workload using the guidelines in the [FSDP README](./fsdp/README.md)

## Prerequisites

- Google Cloud Platform account with billing enabled
- Terraform installed (v1.0.0+)
- Google Cloud SDK
- kubectl configured to work with GKE
- PyTorch 2.0+ (for FSDP support)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- PyTorch team for the FSDP implementation
- Google Cloud Platform documentation
- NVIDIA for T4 GPU optimization guidelines