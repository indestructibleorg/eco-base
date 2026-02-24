# =============================================================================
# Eco-Base Production Infrastructure - GKE Cluster
# =============================================================================
# 企業級雲原生AI平台 - 生產環境基礎設施
# 
# 使用方式:
# 1. 複製 terraform.tfvars.example 為 terraform.tfvars
# 2. 填入您的 GCP 專案 ID 和其他變數
# 3. 執行: terraform init && terraform plan && terraform apply
# =============================================================================

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # 建議使用 GCS 作為 backend 存儲 terraform state
  backend "gcs" {
    bucket = "YOUR_TFSTATE_BUCKET"  # 請替換為您的 GCS bucket 名稱
    prefix = "terraform/production"
  }
}

# =============================================================================
# Provider 配置
# =============================================================================

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "google-beta" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# =============================================================================
# 啟用必要的 GCP API
# =============================================================================

resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# =============================================================================
# VPC 網路配置
# =============================================================================

resource "google_compute_network" "vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "GLOBAL"

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.gcp_region
  network       = google_compute_network.vpc.id

  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }

  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

# =============================================================================
# Cloud Router 和 NAT (用於 Private Nodes 訪問外部網路)
# =============================================================================

resource "google_compute_router" "router" {
  name    = "${var.cluster_name}-router"
  region  = var.gcp_region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.cluster_name}-nat"
  router                             = google_compute_router.router.name
  region                             = var.gcp_region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# =============================================================================
# GKE 集群 - 生產環境配置
# =============================================================================

resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.gcp_region  # 使用區域級集群實現高可用

  # 網路配置
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # 發布版本通道
  release_channel {
    channel = "REGULAR"
  }

  # 最小版本要求 (安全性)
  min_master_version = var.kubernetes_version

  # 刪除默認節點池 (我們將創建自定義節點池)
  remove_default_node_pool = true
  initial_node_count       = 1

  # IP 配置
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # 私有集群配置 (增強安全性)
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false  # 允許公共端點訪問 (可通過授權網路限制)
    master_ipv4_cidr_block  = var.master_cidr
  }

  # 授權網路 (限制訪問 Master API)
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.authorized_networks
      content {
        cidr_block   = cidr_blocks.value.cidr
        display_name = cidr_blocks.value.name
      }
    }
  }

  # 網路策略 (Calico)
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # 工作負載身份 (Workload Identity)
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  # 監控和日誌
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS", "APISERVER", "CONTROLLER_MANAGER", "SCHEDULER"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "APISERVER", "CONTROLLER_MANAGER", "SCHEDULER"]
    managed_prometheus {
      enabled = true
    }
  }

  # 二進制授權 (可選 - 需要 Binary Authorization API)
  # binary_authorization {
  #   evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  # }

  # 維護政策
  maintenance_policy {
    recurring_window {
      start_time = var.maintenance_start_time
      end_time   = var.maintenance_end_time
      recurrence = "FREQ=WEEKLY;BYDAY=SA,SU"
    }
  }

  # 成本管理
  resource_labels = {
    environment = "production"
    managed_by  = "terraform"
    project     = "eco-base"
  }

  depends_on = [
    google_project_service.apis,
    google_compute_router_nat.nat,
  ]

  lifecycle {
    ignore_changes = [initial_node_count]
  }
}

# =============================================================================
# 系統節點池 (運行 kube-system、監控等系統組件)
# =============================================================================

resource "google_container_node_pool" "system" {
  name       = "system-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.primary.name
  node_count = var.system_node_count

  node_config {
    machine_type = var.system_machine_type
    
    # 使用 Container-Optimized OS
    image_type = "COS_CONTAINERD"

    # 工作負載身份
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # 服務帳戶
    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    # 標籤和污點
    labels = {
      "node-type" = "system"
      "workload"  = "infrastructure"
    }

    taint {
      key    = "node-type"
      value  = "system"
      effect = "NO_SCHEDULE"
    }

    # 磁碟配置
    disk_type    = "pd-ssd"
    disk_size_gb = 100

    # 元數據
    metadata = {
      disable-legacy-endpoints = "true"
    }

    tags = ["gke-node", "${var.cluster_name}-system"]
  }

  # 自動修復和自動升級
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # 升級策略
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# =============================================================================
# 通用工作負載節點池
# =============================================================================

resource "google_container_node_pool" "general" {
  name     = "general-pool"
  location = var.gcp_region
  cluster  = google_container_cluster.primary.name

  # 自動擴展配置
  autoscaling {
    min_node_count  = var.general_min_nodes
    max_node_count  = var.general_max_nodes
    location_policy = "BALANCED"
  }

  node_config {
    machine_type = var.general_machine_type
    image_type   = "COS_CONTAINERD"

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = {
      "node-type" = "general"
      "workload"  = "applications"
    }

    disk_type    = "pd-ssd"
    disk_size_gb = 200

    metadata = {
      disable-legacy-endpoints = "true"
    }

    tags = ["gke-node", "${var.cluster_name}-general"]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 2
    max_unavailable = 1
  }
}

# =============================================================================
# AI/GPU 工作負載節點池 (可選)
# =============================================================================

resource "google_container_node_pool" "gpu" {
  count = var.enable_gpu_pool ? 1 : 0

  name     = "gpu-pool"
  location = var.gcp_region
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count  = var.gpu_min_nodes
    max_node_count  = var.gpu_max_nodes
    location_policy = "BALANCED"
  }

  node_config {
    machine_type = var.gpu_machine_type
    image_type   = "COS_CONTAINERD"

    # GPU 配置
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    service_account = google_service_account.gke_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = {
      "node-type" = "gpu"
      "workload"  = "ai-inference"
      "nvidia.com/gpu.present" = "true"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    disk_type    = "pd-ssd"
    disk_size_gb = 500

    metadata = {
      disable-legacy-endpoints = "true"
    }

    tags = ["gke-node", "${var.cluster_name}-gpu"]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# =============================================================================
# Service Account 配置
# =============================================================================

resource "google_service_account" "gke_nodes" {
  account_id   = "${var.cluster_name}-nodes"
  display_name = "GKE Node Service Account"
  description  = "Service account for GKE nodes"
}

# 最小權限 IAM 綁定
resource "google_project_iam_member" "gke_nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/autoscaling.metricsWriter",
    "roles/artifactregistry.reader",
  ])

  project = var.gcp_project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# =============================================================================
# Workload Identity Service Accounts
# =============================================================================

resource "google_service_account" "workloads" {
  for_each = var.workload_service_accounts

  account_id   = "${var.cluster_name}-${each.key}"
  display_name = each.value.display_name
  description  = each.value.description
}

# Workload Identity 綁定
resource "google_service_account_iam_member" "workload_identity" {
  for_each = var.workload_service_accounts

  service_account_id = google_service_account.workloads[each.key].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.gcp_project_id}.svc.id.goog[${each.value.namespace}/${each.value.k8s_service_account}]"
}

# =============================================================================
# Firewall 規則
# =============================================================================

resource "google_compute_firewall" "allow_internal" {
  name        = "${var.cluster_name}-allow-internal"
  network     = google_compute_network.vpc.name
  description = "Allow internal traffic between GKE nodes"

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [var.subnet_cidr, var.pods_cidr, var.services_cidr]
  target_tags   = ["gke-node"]
}

# =============================================================================
# 輸出
# =============================================================================

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "workload_identity_pool" {
  description = "Workload Identity Pool"
  value       = "${var.gcp_project_id}.svc.id.goog"
}

output "get_credentials_command" {
  description = "Command to get cluster credentials"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.gcp_region} --project ${var.gcp_project_id}"
}
