variable "project_id" {
  description = "unique id of gcp project"
  sensitive   = true
}

variable "credentials" {
  description = "service account to be used by gcp"
  sensitive = true
}

variable "region_name" {
  description = "region of project"
  default = "asia-east1"
}

variable "zone_name" {
  description = "zone inside region"
  default = "asia-east1-c"
}

variable "cluster_name" {
  description = "Name of cluster"
  default = "staging"
}

variable "nodepool_name" {
  description = "Name of cluster"
  default = "gpu-pool"
}

variable "node_mach_type" {
  description = "machine type to be used for nodepool"
  default = "n1-standard-32"
}

variable "gpu_type" {
  description = "gpu type to be used in gke k8s cluster"
  default = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "number of gpu to be used in k8s cluster"
  default = 1
}

variable "required_apis" {
  type = list(string)
  default = [
    "container.googleapis.com",
    "compute.googleapis.com",
    "iamcredentials.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "servicemanagement.googleapis.com",
    "serviceusage.googleapis.com",
    "file.googleapis.com"
  ]
}