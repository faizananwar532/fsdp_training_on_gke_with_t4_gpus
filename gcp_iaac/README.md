# GKE Cluster with GPU Support - Terraform Configuration

This repository contains Terraform configuration files to provision a Google Kubernetes Engine (GKE) cluster with GPU support. The setup includes a primary cluster with a dedicated GPU node pool configured with NVIDIA Tesla T4 GPUs.

## Project Structure

```
.
├── main.tf         # Main Terraform configuration file
├── variables.tf    # Variable definitions
├── providers.tf    # Provider configuration
├── backend.tf      # Backend configuration
├── input.tfvars    # Project-specific variable values
└── backend.tfvars  # GCS backend configuration
```

## Files Description

### main.tf
Contains the core resources:
- Enables required Google APIs
- Creates a GKE cluster with Workload Identity enabled
- Creates a GPU node pool with Tesla T4 GPU and proper configuration

### variables.tf
Defines all variables used in the configuration, including:
- Project specific variables (project_id, credentials)
- Region and zone settings
- Cluster and node pool configurations
- GPU specifications
- Required APIs list

### providers.tf
Configures the Google Cloud provider with required version.

### backend.tf
Sets up Google Cloud Storage (GCS) as the Terraform state backend.

### input.tfvars
Contains project-specific variable values:
- Project ID
- Service account credentials path

### backend.tfvars
Contains GCS backend configuration:
- Storage bucket name
- Prefix for state files
- Service account credentials

## Setup Instructions

### Prerequisites
1. Google Cloud Project with billing enabled
2. Service account with appropriate permissions
3. A Google Cloud Storage (GCS) bucket to store Terraform state files
4. Terraform v1.0+ installed
5. Google Cloud SDK installed

### Customization for a New Project

When using this configuration for a different project, replace all occurrences of `beaming-essence-443508-g1` with your new project ID and update all references to service account files.

You need to modify the following files:

#### 1. input.tfvars
```hcl
project_id  = "YOUR_NEW_PROJECT_ID"
credentials = "PATH_TO_YOUR_SERVICE_ACCOUNT_JSON"
```

#### 2. backend.tfvars
```hcl
bucket      = "YOUR_GCS_BUCKET_NAME"
prefix      = "terraform"
credentials = "PATH_TO_YOUR_SERVICE_ACCOUNT_JSON"
```

#### 3. variables.tf (Optional)
You may want to modify the default values of variables such as:
- `region_name` and `zone_name` based on your preferred deployment location
- `cluster_name` and `nodepool_name` to better identify your resources
- `node_mach_type` based on your workload requirements
- `gpu_type` and `gpu_count` based on your computation needs

### Initial Setup

1. **Create a GCS bucket for Terraform state storage**
   ```bash
   gsutil mb gs://YOUR_GCS_BUCKET_NAME
   ```

2. **Configure your Google Cloud project**
   ```bash
   # Set your project ID
   gcloud config set core/project YOUR_PROJECT_ID
   
   # Activate your service account
   gcloud auth activate-service-account --key-file PATH_TO_YOUR_SERVICE_ACCOUNT_JSON
   ```

### Deployment Steps

1. **Initialize Terraform with backend configuration**
   ```bash
   terraform init -backend-config=backend.tfvars
   ```

2. **Plan the deployment**
   ```bash
   terraform plan -var-file=input.tfvars
   ```

3. **Apply the configuration**
   ```bash
   terraform apply -var-file=input.tfvars
   ```

4. **Connect to your new GKE cluster**
   ```bash
   # Configure kubectl to use your new cluster
   gcloud container clusters get-credentials staging --zone <ZONE_NAME> --project YOUR_PROJECT_ID
   
   # Set the kubectl context to your new cluster
   kubectl config use-context gke_<YOUR_PROJECT_ID>_<ZONE_NAME>_staging
   ```

## GPU Notes

This configuration:
- Uses NVIDIA Tesla T4 GPUs by default
- Automatically installs GPU drivers
- Uses COS_CONTAINERD as the node image type which supports GPUs
- Configures Workload Identity for better security

## Security Best Practices

1. Store your `.tfvars` files securely and never commit them to version control
2. Use service accounts with the principle of least privilege
3. Consider using Terraform Cloud or other secure solutions for managing state files
4. Rotate service account keys regularly
5. Set appropriate IAM permissions on your GCS bucket storing Terraform state
6. Use separate service accounts for different environments (dev, staging, prod)

## Cleanup

To destroy all created resources:
```bash
terraform destroy -var-file=input.tfvars
```

## Troubleshooting

If you encounter issues with GPU drivers, check the following:
- Make sure the required APIs are enabled
- Verify that the selected zone supports the GPU type you've specified
- Check GKE node logs for driver installation issues

For connectivity issues:
- Verify network settings and firewall rules
- Ensure that your service account has adequate permissions