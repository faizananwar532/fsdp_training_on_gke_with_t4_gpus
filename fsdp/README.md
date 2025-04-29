# Deploying PyTorch FSDP Training on GKE with T4 GPUs

This guide will walk you through setting up and running your Fully Sharded Data Parallel (FSDP) training on Google Kubernetes Engine (GKE) with T4 GPUs.

## 1. Prerequisites

I can see you've already set up most of the environment. Let's confirm and complete the setup:

- GKE cluster with T4 GPUs is up and running
- NVIDIA drivers are installed on the nodes
- You've authenticated with the cluster

## 2. Deploy NVIDIA Device Plugin (Already Done)

You've already applied the NVIDIA device plugin:
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

And you've verified it's running:
```bash
kubectl get pods -n kube-system | grep nvidia
```

Then create a namespace for your fsdp
```bash
kubectl create namespace training
kubectl config set-context --current --namespace=training
```

## 3. Setting Up PersistentVolume for Data Storage

Create a PersistentVolume and PersistentVolumeClaim for your training data:

```bash
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: standard-rwx

EOF
```

## 4. Create ConfigMap for FSDP Configuration

Apply the FSDP configuration:

```bash
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: fsdp-config
data:
  fsdp_config.yaml: |
    # FSDP Configuration
    fsdp:
      sharding_strategy: FULL_SHARD  # This is Zero-3 equivalent
      cpu_offload: false  # Set to true if you want CPU offloading
      backward_prefetch: BACKWARD_PRE  # Prefetch gradients for backward pass
      forward_prefetch: true  # Prefetch parameters for forward pass
      mixed_precision: true  # Enable mixed precision for T4 GPUs
      activation_checkpointing: false  # Enable if you need activation checkpointing
      communication_param:
        process_group_backend: "nccl"  # Best for GPU communication
      device: "cuda"  # Use CUDA for computation
EOF
```

## 5. Upload Training Scripts to PVC

Create a temporary pod to upload your training scripts to the PVC:

```bash
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: data-uploader
spec:
  containers:
  - name: data-uploader
    image: python:3.9-slim
    command: ["sleep", "3600"]
    volumeMounts:
    - name: training-data
      mountPath: /data
    securityContext:
      runAsUser: 0  # ✅ run as root user
  volumes:
  - name: training-data
    persistentVolumeClaim:
      claimName: training-data-pvc
EOF

```

Wait for the pod to be running:

```bash
kubectl wait --for=condition=Ready pod/data-uploader

# Install NumPy using pip
kubectl exec -it data-uploader -- pip install numpy
```

Copy your training scripts to the PVC:

```bash
cd fsdp
# Copy the Python files to the pod
kubectl cp fsdp_trainer.py data-uploader:/data/
kubectl cp optimized_dataloader.py data-uploader:/data/
kubectl cp fsdp_config.yaml data-uploader:/data/

# Create directories for the dataset inside the pod
kubectl exec -it data-uploader -- mkdir -p /data/train /data/val

# Create dummy data inside the pod
cat << EOF > create_dummy_data.py
import numpy as np
import os

# Create train data
os.makedirs('/data/train', exist_ok=True)
np.save('/data/train/features.npy', np.random.randn(10000, 1024).astype(np.float32))
np.save('/data/train/labels.npy', np.random.randint(0, 10, size=10000))

# Create validation data
os.makedirs('/data/val', exist_ok=True)
np.save('/data/val/features.npy', np.random.randn(2000, 1024).astype(np.float32))
np.save('/data/val/labels.npy', np.random.randint(0, 10, size=2000))

print("Dummy data created successfully!")
EOF

# Copy and execute
kubectl cp create_dummy_data.py data-uploader:/data/
kubectl exec -it data-uploader -- python /data/create_dummy_data.py

# Delete the uploader pod when done
kubectl delete pod data-uploader
```

## 6. Deploy Headless Service for StatefulSet Communication

```bash
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: fsdp-training
spec:
  clusterIP: None  # Headless service for StatefulSet
  selector:
    app: fsdp-training
  ports:
  - port: 29500
    name: nccl
EOF
```

## 7. Deploy FSDP Training StatefulSet

Now apply the full deployment:

```bash
kubectl apply -f fsdp_deployment.yaml
```

## 8. Monitor Training Progress

Watch the pods as they start up:

```bash
kubectl get statefulset
kubectl get pods -w
```

Check logs from the master node (pod 0):

```bash
kubectl logs -f fsdp-training-0
```

Check logs from worker node (pod 1):

```bash
kubectl logs -f fsdp-training-1
```

## 9. Deploying Using Kubeflow Pipelines (Optional)

If you want to use Kubeflow Pipelines for orchestration instead of manual deployment:

1. Make sure your Kubeflow installation is up and running
2. Compile the pipeline:

```bash
python fsdp_kubeflow_pipeline.py
```

3. Upload the generated `fsdp_training_pipeline.yaml` to your Kubeflow Pipelines UI
4. Create a run with appropriate parameters:
   - `input_path`: "/mnt/data"
   - `output_path`: "/mnt/output"
   - `num_gpus`: 2 (or however many T4 GPUs you have)

## 10. Visualizing Training Progress (Optional)

To set up a TensorBoard instance to monitor training:

```bash
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: tensorboard
spec:
  containers:
  - name: tensorboard
    image: tensorflow/tensorflow:2.8.0
    command:
    - tensorboard
    - --logdir=/output
    - --port=6006
    volumeMounts:
    - name: output-data
      mountPath: /output
    ports:
    - containerPort: 6006
  volumes:
  - name: output-data
    persistentVolumeClaim:
      claimName: training-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: tensorboard
spec:
  selector:
    app: tensorboard
  ports:
  - port: 80
    targetPort: 6006
  type: LoadBalancer
EOF
```

## 11. Retrieving Training Results

After training completes, create a pod to retrieve the results:

```bash
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: results-downloader
spec:
  containers:
  - name: downloader
    image: ubuntu:20.04
    command: ["sleep", "3600"]
    volumeMounts:
    - name: output-data
      mountPath: /output
  volumes:
  - name: output-data
    persistentVolumeClaim:
      claimName: training-data-pvc
EOF
```

Then use kubectl cp to download the results:

```bash
kubectl cp results-downloader:/output/model_best.pt ./model_best.pt
```

## 12. Clean Up Resources When Done

```bash
# Delete the StatefulSet
kubectl delete statefulset fsdp-training

# Delete the Service
kubectl delete service fsdp-training

# Delete PVCs if no longer needed
kubectl delete pvc training-data-pvc

# Delete ConfigMap
kubectl delete configmap fsdp-config
```

## Troubleshooting

### Common Issues

1. **Pods Stuck in Pending State**: Check if there are enough T4 GPUs available in your cluster:
   ```bash
   kubectl describe pod fsdp-training-0
   ```

2. **NCCL Communication Issues**: Check NCCL debug logs:
   ```bash
   kubectl logs fsdp-training-0 | grep NCCL
   ```

3. **OOM Errors**: Consider enabling activation checkpointing in your FSDP config:
   ```yaml
   activation_checkpointing: true
   ```

4. **Slow Training**: Ensure mixed precision is enabled and appropriate batch sizes for T4 GPUs:
   ```yaml
   mixed_precision: true
   ```

5. **StatefulSet Pods Not Communicating**: Verify the headless service is working correctly:
   ```bash
   kubectl describe service fsdp-training
   ```

   