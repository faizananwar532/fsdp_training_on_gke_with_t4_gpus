import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from kubernetes import client as k8s_client

# Define function for preprocessing data
@dsl.component
def preprocess_data(input_path: str, output_path: str):
    import os
    import torch
    import numpy as np
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    
    # This is where you would implement your actual data preprocessing
    # For demonstration purposes, we're creating dummy data
    print(f"Preprocessing data from {input_path} to {output_path}")
    
    # Example preprocessing logic (replace with your actual preprocessing)
    # Simulating creation of preprocessed files with float32 data for T4 GPUs
    np.save(os.path.join(output_path, 'train/features.npy'), np.random.randn(10000, 1024).astype(np.float32))
    np.save(os.path.join(output_path, 'train/labels.npy'), np.random.randint(0, 10, size=10000))
    np.save(os.path.join(output_path, 'val/features.npy'), np.random.randn(2000, 1024).astype(np.float32))
    np.save(os.path.join(output_path, 'val/labels.npy'), np.random.randint(0, 10, size=2000))
    
    # Also copy the trainer scripts to the output directory so they're accessible to the training pods
    with open(os.path.join(output_path, 'fsdp_trainer.py'), 'w') as f:
        with open('/fsdp_trainer.py', 'r') as src:
            f.write(src.read())
    
    with open(os.path.join(output_path, 'optimized_dataloader.py'), 'w') as f:
        with open('/optimized_dataloader.py', 'r') as src:
            f.write(src.read())
    
    print("Data preprocessing completed.")
    return output_path

# Define function to create ConfigMap for FSDP config
@dsl.component
def create_fsdp_config():
    import yaml
    from kubernetes import client, config
    
    # Load in-cluster configuration
    config.load_incluster_config()
    
    # Define FSDP config
    fsdp_config = {
        "fsdp": {
            "sharding_strategy": "FULL_SHARD",  # Zero-3 equivalent
            "cpu_offload": False,  # Set to true if you want CPU offloading
            "backward_prefetch": "BACKWARD_PRE",  # Prefetch gradients for backward pass
            "forward_prefetch": True,  # Prefetch parameters for forward pass
            "mixed_precision": True,  # Enable mixed precision for T4 GPUs
            "activation_checkpointing": False,  # Enable if you need activation checkpointing
            "communication_param": {
                "process_group_backend": "nccl"  # Best for GPU communication
            },
            "device": "cuda"  # Use CUDA for computation
        }
    }
    
    # Create ConfigMap
    api = client.CoreV1Api()
    
    # Convert config to YAML
    config_yaml = yaml.dump(fsdp_config)
    
    # Define ConfigMap
    configmap = client.V1ConfigMap(
        api_version="v1",
        kind="ConfigMap",
        metadata=client.V1ObjectMeta(name="fsdp-config"),
        data={"fsdp_config.yaml": config_yaml}
    )
    
    # Create ConfigMap
    try:
        api.create_namespaced_config_map(namespace="default", body=configmap)
        print("ConfigMap created successfully")
    except client.rest.ApiException as e:
        if e.status == 409:  # Already exists
            api.patch_namespaced_config_map(name="fsdp-config", namespace="default", body=configmap)
            print("ConfigMap updated successfully")
        else:
            raise e
    
    return "fsdp-config"

# Define function to deploy FSDP training StatefulSet
@dsl.component
def deploy_fsdp_training(data_path: str, num_gpus: int = 2):
    from kubernetes import client, config
    import yaml
    import time
    
    # Load in-cluster configuration
    config.load_incluster_config()
    
    # Create Kubernetes API clients
    apps_api = client.AppsV1Api()
    core_api = client.CoreV1Api()
    
    # Create headless service for StatefulSet
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name="fsdp-training"),
        spec=client.V1ServiceSpec(
            cluster_ip="None",  # Headless service
            selector={"app": "fsdp-training"},
            ports=[client.V1ServicePort(port=29500, name="nccl")]
        )
    )
    
    try:
        core_api.create_namespaced_service(namespace="default", body=service)
        print("Service created successfully")
    except client.rest.ApiException as e:
        if e.status == 409:  # Already exists
            print("Service already exists")
        else:
            raise e
    
    # Create PVC for training data
    pvc = client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=client.V1ObjectMeta(name="training-data-pvc"),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteMany"],
            resources=client.V1ResourceRequirements(
                requests={"storage": "100Gi"}
            )
        )
    )
    
    try:
        core_api.create_namespaced_persistent_volume_claim(namespace="default", body=pvc)
        print("PVC created successfully")
    except client.rest.ApiException as e:
        if e.status == 409:  # Already exists
            print("PVC already exists")
        else:
            raise e
    
    # Define StatefulSet
    stateful_set = client.V1StatefulSet(
        api_version="apps/v1",
        kind="StatefulSet",
        metadata=client.V1ObjectMeta(name="fsdp-training"),
        spec=client.V1StatefulSetSpec(
            service_name="fsdp-training",
            replicas=num_gpus,
            selector=client.V1LabelSelector(
                match_labels={"app": "fsdp-training"}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": "fsdp-training"}
                ),
                spec=client.V1PodSpec(
                    termination_grace_period_seconds=30,
                    affinity=client.V1Affinity(
                        node_affinity=client.V1NodeAffinity(
                            required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                                node_selector_terms=[
                                    client.V1NodeSelectorTerm(
                                        match_expressions=[
                                            client.V1NodeSelectorRequirement(
                                                key="cloud.google.com/gke-accelerator",
                                                operator="In",
                                                values=["nvidia-tesla-t4"]
                                            )
                                        ]
                                    )
                                ]
                            )
                        )
                    ),
                    containers=[
                        client.V1Container(
                            name="pytorch-fsdp",
                            image="nvcr.io/nvidia/pytorch:23.01-py3",
                            resources=client.V1ResourceRequirements(
                                limits={
                                    "nvidia.com/gpu": 1,
                                    "cpu": "30",
                                    "memory": "110Gi"
                                },
                                requests={
                                    "nvidia.com/gpu": 1,
                                    "cpu": "24",
                                    "memory": "90Gi"
                                }
                            ),
                            volume_mounts=[
                                client.V1VolumeMount(name="training-data", mount_path="/data"),
                                client.V1VolumeMount(name="output-data", mount_path="/output"),
                                client.V1VolumeMount(name="fsdp-config-volume", mount_path="/etc/fsdp-config"),
                                client.V1VolumeMount(name="shared-memory", mount_path="/dev/shm"),
                                client.V1VolumeMount(name="dshm", mount_path="/dev/shm")
                            ],
                            env=[
                                client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
                                client.V1EnvVar(name="OMP_NUM_THREADS", value="16"),
                                client.V1EnvVar(name="NCCL_DEBUG", value="INFO"),
                                client.V1EnvVar(name="NCCL_IB_DISABLE", value="0"),
                                client.V1EnvVar(name="NCCL_SOCKET_IFNAME", value="eth0"),
                                client.V1EnvVar(name="MASTER_ADDR", value="fsdp-training-0.fsdp-training.default.svc.cluster.local"),
                                client.V1EnvVar(name="MASTER_PORT", value="29500"),
                                client.V1EnvVar(name="WORLD_SIZE", value=str(num_gpus)),
                                client.V1EnvVar(
                                    name="HOSTNAME",
                                    value_from=client.V1EnvVarSource(
                                        field_ref=client.V1ObjectFieldSelector(
                                            field_path="metadata.name"
                                        )
                                    )
                                ),
                                client.V1EnvVar(
                                    name="NODE_RANK",
                                    value_from=client.V1EnvVarSource(
                                        field_ref=client.V1ObjectFieldSelector(
                                            field_path="metadata.name"
                                        )
                                    )
                                )
                            ],
                            command=["/bin/bash", "-c"],
                            args=[
                                """
                                # Extract pod index from hostname for NODE_RANK
                                IFS='-' read -ra ADDR <<< "$HOSTNAME"
                                export NODE_RANK=${ADDR[-1]}
                                
                                # Install required packages
                                pip install pyyaml
                                
                                # Copy training scripts
                                echo "Copying training scripts..."
                                cp /data/fsdp_trainer.py /workspace/
                                cp /data/optimized_dataloader.py /workspace/
                                
                                echo "Starting training with NODE_RANK=$NODE_RANK, WORLD_SIZE=$WORLD_SIZE"
                                cd /workspace
                                python fsdp_trainer.py \\
                                  --config /etc/fsdp-config/fsdp_config.yaml \\
                                  --data-path /data \\
                                  --output-path /output \\
                                  --node-rank $NODE_RANK \\
                                  --batch-size 64 \\
                                  --epochs 10 \\
                                  --lr 0.001
                                """
                            ]
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="training-data",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name="training-data-pvc"
                            )
                        ),
                        client.V1Volume(
                            name="output-data",
                            empty_dir=client.V1EmptyDirVolumeSource()
                        ),
                        client.V1Volume(
                            name="fsdp-config-volume",
                            config_map=client.V1ConfigMapVolumeSource(
                                name="fsdp-config"
                            )
                        ),
                        client.V1Volume(
                            name="shared-memory",
                            empty_dir=client.V1EmptyDirVolumeSource(
                                medium="Memory"
                            )
                        ),
                        client.V1Volume(
                            name="dshm",
                            empty_dir=client.V1EmptyDirVolumeSource(
                                medium="Memory",
                                size_limit="16Gi"
                            )
                        )
                    ]
                )
            )
        )
    )
    
    # Create StatefulSet
    try:
        apps_api.create_namespaced_stateful_set(namespace="default", body=stateful_set)
        print("StatefulSet created successfully")
    except client.rest.ApiException as e:
        if e.status == 409:  # Already exists
            apps_api.delete_namespaced_stateful_set(name="fsdp-training", namespace="default")
            print("Existing StatefulSet deleted")
            time.sleep(10)  # Wait for deletion
            apps_api.create_namespaced_stateful_set(namespace="default", body=stateful_set)
            print("StatefulSet recreated successfully")
        else:
            raise e
    
    # Wait for pods to start
    print("Waiting for pods to start...")
    time.sleep(30)
    
    return "fsdp-training"

# Define function to monitor training progress
@dsl.component
def monitor_training(stateful_set_name: str, timeout_seconds: int = 1800):
    from kubernetes import client, config, watch
    import time
    
    # Load in-cluster configuration
    config.load_incluster_config()
    
    # Create Kubernetes API clients
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()
    
    # Get StatefulSet
    stateful_set = apps_api.read_namespaced_stateful_set(name=stateful_set_name, namespace="default")
    replicas = stateful_set.spec.replicas
    
    print(f"Monitoring training with {replicas} replicas")
    
    # Wait for all pods to be ready
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        pod_list = core_api.list_namespaced_pod(
            namespace="default",
            label_selector=f"app={stateful_set_name}"
        )
        
        ready_pods = 0
        for pod in pod_list.items:
            if pod.status.phase == "Running":
                container_statuses = pod.status.container_statuses
                if container_statuses and container_statuses[0].ready:
                    ready_pods += 1
        
        if ready_pods == replicas:
            print(f"All {replicas} pods are ready")
            break
        
        print(f"{ready_pods}/{replicas} pods are ready")
        time.sleep(10)
    
    # Stream logs from the first pod
    try:
        logs = core_api.read_namespaced_pod_log(
            name=f"{stateful_set_name}-0",
            namespace="default",
            container="pytorch-fsdp",
            follow=False,
            tail_lines=100
        )
        print("Recent logs from master pod:")
        print(logs)
    except Exception as e:
        print(f"Error getting logs: {e}")
    
    # Check training status
    all_pods = core_api.list_namespaced_pod(
        namespace="default",
        label_selector=f"app={stateful_set_name}"
    )
    
    # Return status summary
    statuses = []
    for pod in all_pods.items:
        statuses.append(f"{pod.metadata.name}: {pod.status.phase}")
    
    return "\n".join(statuses)

# Define the pipeline
@dsl.pipeline(
    name="FSDP Training Pipeline",
    description="Pipeline for FSDP training on GKE with T4 GPUs"
)
def fsdp_training_pipeline(
    input_path: str = "/mnt/data",
    output_path: str = "/mnt/output",
    num_gpus: int = 2
):
    # Step 1: Preprocess data
    preprocess_op = preprocess_data(input_path=input_path, output_path=output_path)
    
    # Step 2: Create FSDP config
    config_op = create_fsdp_config().after(preprocess_op)
    
    # Step 3: Deploy FSDP training
    deploy_op = deploy_fsdp_training(
        data_path=preprocess_op.output,
        num_gpus=num_gpus
    ).after(config_op)
    
    # Step 4: Monitor training
    monitor_op = monitor_training(
        stateful_set_name=deploy_op.output,
        timeout_seconds=3600  # 1 hour timeout
    )

# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=fsdp_training_pipeline,
        package_path="fsdp_training_pipeline.yaml"
    )
    
    print("Pipeline compiled successfully to fsdp_training_pipeline.yaml")
    print("You can now upload this pipeline to your Kubeflow installation")