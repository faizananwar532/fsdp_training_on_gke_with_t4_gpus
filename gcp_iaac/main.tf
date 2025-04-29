resource "google_project_service" "required" {
  for_each = toset(var.required_apis)
  project = var.project_id
  service = each.key
  disable_on_destroy = false
}

resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone_name
  
  remove_default_node_pool = true
  deletion_protection      = false
  initial_node_count       = 2
  
  # Enable Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Fix GPU support configuration 
  addons_config {
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }
  
  lifecycle {
    ignore_changes = [enable_l4_ilb_subsetting]
  }
  
  depends_on = [google_project_service.required]
}

resource "google_container_node_pool" "gpu_node_pool" {
  name       = var.nodepool_name
  cluster    = google_container_cluster.primary.name
  location   = google_container_cluster.primary.location
  initial_node_count = 1
  
  node_config {
    machine_type = var.node_mach_type
    disk_size_gb = 300  
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Configure the T4 GPU
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    
    # Now this will work because we've enabled Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Use a GKE image with GPU support
    image_type = "COS_CONTAINERD"
    
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
  
  # This tells GKE to install GPU drivers automatically
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  depends_on = [google_container_cluster.primary]
}