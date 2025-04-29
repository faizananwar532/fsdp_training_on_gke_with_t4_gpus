terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "6.32.0"
    }
  }
}

provider "google" {
  # Configuration options
    credentials = var.credentials
    project = var.project_id 
    region  = var.region_name
    zone    = var.zone_name
}

