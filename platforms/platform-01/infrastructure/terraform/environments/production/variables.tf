# =============================================================================
# 變數定義 - Production Environment
# =============================================================================

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
  # 範例: "my-project-ops-1991"
}

variable "gcp_region" {
  description = "GCP Region for the cluster"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "eco-base-production"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

# =============================================================================
# 網路配置
# =============================================================================

variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  description = "CIDR range for pods"
  type        = string
  default     = "10.4.0.0/14"
}

variable "services_cidr" {
  description = "CIDR range for services"
  type        = string
  default     = "10.8.0.0/20"
}

variable "master_cidr" {
  description = "CIDR range for master nodes"
  type        = string
  default     = "172.16.0.0/28"
}

variable "authorized_networks" {
  description = "List of authorized networks to access the cluster"
  type = list(object({
    name = string
    cidr = string
  }))
  default = []
  # 範例:
  # [
  #   { name = "office", cidr = "203.0.113.0/24" },
  #   { name = "vpn", cidr = "198.51.100.0/24" }
  # ]
}

# =============================================================================
# 節點池配置
# =============================================================================

variable "system_node_count" {
  description = "Number of nodes in system pool"
  type        = number
  default     = 2
}

variable "system_machine_type" {
  description = "Machine type for system nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "general_min_nodes" {
  description = "Minimum number of general nodes"
  type        = number
  default     = 2
}

variable "general_max_nodes" {
  description = "Maximum number of general nodes"
  type        = number
  default     = 10
}

variable "general_machine_type" {
  description = "Machine type for general nodes"
  type        = string
  default     = "e2-standard-8"
}

# =============================================================================
# GPU 節點池配置
# =============================================================================

variable "enable_gpu_pool" {
  description = "Enable GPU node pool for AI workloads"
  type        = bool
  default     = true
}

variable "gpu_min_nodes" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_max_nodes" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 4
}

variable "gpu_machine_type" {
  description = "Machine type for GPU nodes"
  type        = string
  default     = "n1-standard-8"
}

variable "gpu_type" {
  description = "Type of GPU"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

# =============================================================================
# 維護窗口配置
# =============================================================================

variable "maintenance_start_time" {
  description = "Maintenance window start time (UTC)"
  type        = string
  default     = "2024-01-01T02:00:00Z"  # 每週六凌晨 2 點 UTC
}

variable "maintenance_end_time" {
  description = "Maintenance window end time (UTC)"
  type        = string
  default     = "2024-01-01T06:00:00Z"  # 每週日凌晨 6 點 UTC
}

# =============================================================================
# Workload Identity Service Accounts
# =============================================================================

variable "workload_service_accounts" {
  description = "Map of workload service accounts to create"
  type = map(object({
    display_name        = string
    description         = string
    namespace           = string
    k8s_service_account = string
  }))
  default = {
    "api" = {
      display_name        = "API Service Account"
      description         = "Service account for API workloads"
      namespace           = "production"
      k8s_service_account = "api-sa"
    }
    "ai" = {
      display_name        = "AI Service Account"
      description         = "Service account for AI inference workloads"
      namespace           = "production"
      k8s_service_account = "ai-sa"
    }
    "argo" = {
      display_name        = "ArgoCD Service Account"
      description         = "Service account for ArgoCD"
      namespace           = "argocd"
      k8s_service_account = "argocd-application-controller"
    }
  }
}
