# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: documentation
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
# Enterprise Infrastructure Deployment Scripts

This directory contains enterprise-grade deployment scripts for testing the MachineNativeOps infrastructure components.

## Scripts Overview

### 1. deploy-istio-testing.sh
Deploys Istio Service Mesh to the testing environment.

**Features:**
- Creates testing namespace with sidecar injection
- Deploys Istio gateway and virtual services
- Applies security policies (mTLS, RBAC)
- Configures traffic management rules
- Deploys sample application for testing
- Verifies Istio configuration

**Usage:**
```bash
./scripts/deploy-istio-testing.sh
```

**What it does:**
1. Checks prerequisites (kubectl, istioctl)
2. Creates namespace `machine-native-ops-testing`
3. Enables Istio sidecar injection
4. Deploys Istio configuration (gateway, virtual services, destination rules)
5. Applies security policies
6. Deploys sample application
7. Tests Istio routing
8. Verifies all components

**Verification:**
```bash
# Check namespace
kubectl get namespace machine-native-ops-testing

# Check Istio resources
kubectl get all -n machine-native-ops-testing

# Check sidecar injection
kubectl describe pod <pod-name> -n machine-native-ops-testing | grep -i istio
```

---

### 2. install-jaeger-testing.sh
Installs Jaeger distributed tracing with Elasticsearch backend.

**Features:**
- Installs Elasticsearch Operator
- Deploys Elasticsearch cluster
- Installs Jaeger with Elasticsearch backend
- Configures Istio for Jaeger tracing
- Deploys tracing demo application
- Generates test traffic
- Verifies trace collection
- Checks Jaeger UI access

**Usage:**
```bash
./scripts/install-jaeger-testing.sh
```

**What it does:**
1. Installs Elasticsearch Operator
2. Deploys Elasticsearch (1 node, 20Gi storage)
3. Installs Jaeger (agent, collector, query)
4. Configures Istio telemetry for tracing
5. Deploys HotROD demo application
6. Generates test traffic
7. Verifies Jaeger installation
8. Provides Jaeger UI access instructions

**Access Jaeger UI:**
```bash
kubectl port-forward -n jaeger svc/jaeger-query 16686:16686
# Open browser at [EXTERNAL_URL_REMOVED]
```

**Verification:**
```bash
# Check Jaeger pods
kubectl get pods -n jaeger

# Check Elasticsearch
kubectl get elasticsearch -n jaeger

# Check Jaeger services
kubectl get svc -n jaeger

# Test tracing
kubectl logs -n jaeger deployment/jaeger-collector
```

---

### 3. configure-velero-backups.sh
Configures Velero for automated backups in testing environment.

**Features:**
- Installs Velero CLI (if not installed)
- Configures S3 credentials
- Installs Velero server
- Deploys backup schedules
- Creates backup jobs
- Creates manual backup
- Tests restore operation
- Verifies Velero installation

**Usage:**
```bash
# Set S3 credentials first
export S3_ENDPOINT="[EXTERNAL_URL_REMOVED]
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"

# Run script
./scripts/configure-velero-backups.sh
```

**What it does:**
1. Checks and installs Velero CLI
2. Configures S3 credentials
3. Installs Velero server with AWS provider
4. Creates backup schedules (daily, hourly, config)
5. Deploys backup validation job
6. Creates manual backup
7. Tests restore operation
8. Verifies all components

**Backup Schedules:**
- **Daily Backup:** 2 AM daily, 7 days retention
- **Hourly Backup:** Every hour, 1 day retention
- **Config Backup:** Every 30 minutes, 3 days retention

**Verification:**
```bash
# Check Velero pods
kubectl get pods -n velero

# Check backups
velero backup get

# Check schedules
kubectl get schedule -n velero

# Check backup locations
velero backup-location get

# Test restore
velero restore create --from-backup <backup-name>
```

---

### 4. deploy-multi-az-resources.sh
Deploys multi-AZ resources with zone-aware routing.

**Features:**
- Labels nodes with zone information
- Deploys multi-AZ application (9 replicas, 3 per zone)
- Configures zone-aware routing
- Sets up Pod Disruption Budgets
- Configures HPA for auto-scaling
- Creates Priority Classes
- Tests load balancing
- Verifies pod distribution

**Usage:**
```bash
./scripts/deploy-multi-az-resources.sh
```

**What it does:**
1. Labels nodes with zone topology
2. Deploys multi-AZ application
3. Applies topology spread constraints
4. Configures zone-aware routing (80% local, 20% cross-zone)
5. Sets up Pod Disruption Budget (66% min available)
6. Configures HPA (9-15 replicas)
7. Creates Priority Classes
8. Tests load balancing
9. Verifies pod distribution

**Zone Distribution:**
- **us-east-1a:** 3 replicas
- **us-east-1b:** 3 replicas
- **us-east-1c:** 3 replicas

**Verification:**
```bash
# Check pod distribution
kubectl get pods -n machine-native-ops-testing -o wide

# Check node labels
kubectl get nodes -L topology.kubernetes.io/zone

# Check HPA
kubectl get hpa -n machine-native-ops-testing

# Check PDB
kubectl get pdb -n machine-native-ops-testing

# Check zone-aware routing
istioctl pc destinationrule <pod-name>
```

---

### 5. test-failover-scenarios.sh
Tests failover scenarios for multi-AZ deployment.

**Features:**
- Tests pod failover
- Simulates zone failure
- Tests database failover
- Verifies circuit breaker
- Tests automatic recovery
- Generates detailed test report

**Usage:**
```bash
./scripts/test-failover-scenarios.sh
```

**What it does:**
1. Initializes test results
2. Tests pod failover (delete pod, verify recreation)
3. Simulates zone failure (cordon nodes, verify redistribution)
4. Tests database failover (delete primary, verify promotion)
5. Verifies circuit breaker configuration
6. Tests automatic recovery (delete multiple pods)
7. Generates detailed test report

**Test Results:**
```bash
# View test report
cat /tmp/failover-test-results/test-report.md

# Check pod status
kubectl get pods -n machine-native-ops-testing

# Check pod distribution
kubectl get pods -n machine-native-ops-testing -o wide
```

---

## Deployment Order

Deploy the components in the following order:

1. **Deploy Istio:**
   ```bash
   ./scripts/deploy-istio-testing.sh
   ```

2. **Install Jaeger:**
   ```bash
   ./scripts/install-jaeger-testing.sh
   ```

3. **Configure Velero:**
   ```bash
   export S3_ENDPOINT="[EXTERNAL_URL_REMOVED]
   export S3_ACCESS_KEY="your-access-key"
   export S3_SECRET_KEY="your-secret-key"
   ./scripts/configure-velero-backups.sh
   ```

4. **Deploy Multi-AZ Resources:**
   ```bash
   ./scripts/deploy-multi-az-resources.sh
   ```

5. **Test Failover Scenarios:**
   ```bash
   ./scripts/test-failover-scenarios.sh
   ```

---

## Prerequisites

All scripts require:
- **kubectl**: Kubernetes CLI
- **Helm**: For Jaeger and Velero installation
- **istioctl**: Istio CLI (for Istio deployment)
- **Kubernetes Cluster**: With 3+ nodes for multi-AZ testing
- **S3 Storage**: For Velero backups

---

## Configuration

### Environment Variables

**For Velero:**
```bash
export S3_ENDPOINT="[EXTERNAL_URL_REMOVED]
export S3_ACCESS_KEY="your-access-key"
export S3_SECRET_KEY="your-secret-key"
```

### Script Configuration

Each script has configurable variables at the top:

```bash
NAMESPACE="machine-native-ops-testing"
ZONES=("us-east-1a" "us-east-1b" "us-east-1c")
```

Modify these as needed for your environment.

---

## Troubleshooting

### Istio Issues

**Sidecar not injected:**
```bash
# Check namespace label
kubectl get namespace machine-native-ops-testing -o yaml | grep istio-injection

# Re-enable injection
kubectl label namespace machine-native-ops-testing istio-injection=enabled --overwrite
```

**Gateway not accessible:**
```bash
# Check gateway status
kubectl get gateway -n machine-native-ops-testing

# Check gateway pods
kubectl get pods -n istio-system -l istio=ingressgateway
```

### Jaeger Issues

**No traces appearing:**
```bash
# Check Jaeger collector logs
kubectl logs -n jaeger deployment/jaeger-collector

# Check Istio telemetry
kubectl get telemetry -n istio-system

# Verify sampling rate
kubectl get telemetry jaeger-tracing-testing -n istio-system -o yaml | grep randomSamplingPercentage
```

**Elasticsearch connection issues:**
```bash
# Check Elasticsearch status
kubectl get elasticsearch -n jaeger

# Check Elasticsearch pods
kubectl get pods -n jaeger -l elasticsearch.k8s.elastic.co/cluster-name=jaeger-es

# Check Elasticsearch logs
kubectl logs -n jaeger -l elasticsearch.k8s.elastic.co/cluster-name=jaeger-es
```

### Velero Issues

**Backup failures:**
```bash
# Check Velero logs
kubectl logs -n velero deployment/velero

# Check backup status
velero backup get --details

# Check backup location
velero backup-location get --show-details

# Test S3 connectivity
aws s3 ls s3://machine-native-ops-backups-testing/
```

**Restore failures:**
```bash
# Check restore status
velero restore get --details

# Check restore logs
kubectl logs -n velero deployment/velero | grep restore

# Verify backup integrity
velero backup describe <backup-name> --details
```

### Multi-AZ Issues

**Pods not distributed:**
```bash
# Check node labels
kubectl get nodes -L topology.kubernetes.io/zone

# Check topology constraints
kubectl describe deployment multi-az-app -n machine-native-ops-testing | grep topologySpread

# Force pod redistribution
kubectl delete pods -n machine-native-ops-testing -l app=multi-az-app
```

**Zone routing not working:**
```bash
# Check DestinationRule
kubectl get destinationrule -n machine-native-ops-testing

# Check zone-aware routing config
kubectl get destinationrule multi-az-app-dr -n machine-native-ops-testing -o yaml | grep locality

# Verify with istioctl
istioctl pc destinationrule <pod-name> -n machine-native-ops-testing
```

---

## Cleanup

To remove all deployed resources:

```bash
# Remove Istio resources
kubectl delete namespace machine-native-ops-testing

# Remove Jaeger
helm uninstall jaeger -n jaeger
kubectl delete namespace jaeger
helm uninstall elastic-operator -n elasticsearch
kubectl delete namespace elasticsearch

# Remove Velero
velero uninstall
kubectl delete namespace velero

# Remove backups from S3
aws s3 rm s3://machine-native-ops-backups-testing/ --recursive
```

---

## Best Practices

1. **Run scripts in order**: Follow the deployment order above
2. **Verify each step**: Check logs and pod status after each script
3. **Test incrementally**: Test each component before moving to the next
4. **Monitor resources**: Keep an eye on resource usage during deployment
5. **Review logs**: Check logs for errors and warnings
6. **Document changes**: Keep track of any customizations made

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review script logs
3. Check Kubernetes events: `kubectl get events -n <namespace>`
4. Check pod logs: `kubectl logs <pod-name> -n <namespace>`
5. Review the main documentation in `k8s/` directory

---

## Next Steps

After successful deployment:

1. **Monitor the environment**: Set up monitoring and alerting
2. **Test additional scenarios**: Add more failover tests
3. **Optimize configuration**: Adjust resource limits and scaling parameters
4. **Document procedures**: Create operational runbooks
5. **Prepare for production**: Review and adjust configurations for production use

---

**Last Updated:** $(date)
**Version:** 1.0.0