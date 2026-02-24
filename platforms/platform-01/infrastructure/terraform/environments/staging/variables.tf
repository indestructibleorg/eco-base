variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "eco-base-staging"
}

variable "subnet_cidr" {
  description = "Subnet CIDR"
  type        = string
  default     = "10.16.0.0/20"
}

variable "pods_cidr" {
  description = "Pods CIDR"
  type        = string
  default     = "10.20.0.0/14"
}

variable "services_cidr" {
  description = "Services CIDR"
  type        = string
  default     = "10.24.0.0/20"
}

variable "master_cidr" {
  description = "Master CIDR"
  type        = string
  default     = "172.16.16.0/28"
}

variable "machine_type" {
  description = "Node machine type"
  type        = string
  default     = "e2-standard-4"
}
