#!/bin/bash
# Supply Chain Tools Setup Script for MachineNativeOps
# This script installs and configures supply chain security tools

set -e

echo "=== Supply Chain Tools Setup for MachineNativeOps ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning message
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_warning "Some operations may require sudo privileges"
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p workspace/artifacts/{sbom,provenance,signatures,rekor,security,compliance}
mkdir -p workspace/tools
print_success "Directory structure created"

# Install syft (SBOM generation)
echo ""
echo "Installing syft..."
if ! command -v syft &> /dev/null; then
    wget -qO- https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    print_success "syft installed"
else
    print_success "syft already installed"
fi

# Install trivy (vulnerability scanning)
echo ""
echo "Installing trivy..."
if ! command -v trivy &> /dev/null; then
    wget -qO- https://raw.githubusercontent.com/aquasecurity/trivy/main/install.sh | sh -s -- -b /usr/local/bin
    print_success "trivy installed"
else
    print_success "trivy already installed"
fi

# Install cosign (artifact signing)
echo ""
echo "Installing cosign..."
if ! command -v cosign &> /dev/null; then
    wget -qO- https://raw.githubusercontent.com/sigstore/cosign/main/install.sh | sh -s -- -b /usr/local/bin
    print_success "cosign installed"
else
    print_success "cosign already installed"
fi

# Install OPA (policy engine)
echo ""
echo "Installing OPA..."
if ! command -v opa &> /dev/null; then
    wget -qO- https://raw.githubusercontent.com/open-policy-agent/opa/main/install.sh | sh -s -- -b /usr/local/bin
    print_success "OPA installed"
else
    print_success "OPA already installed"
fi

# Create configuration files
echo ""
echo "Creating configuration files..."

# Create cosign config
cat > workspace/tools/cosign-config.yaml << 'EOF'
# Cosign Configuration
sign-blob:
  annotations:
    "mno.org/component": "machine-native-ops"
    "mno.org/environment": "production"

verify-blob:
  certificate-identity: "https://github.com/MachineNativeOps/machine-native-ops/.github/workflows/supply-chain-security.yml@refs/heads/main"
  certificate-oidc-issuer: "https://token.actions.githubusercontent.com"
EOF

print_success "Cosign configuration created"

# Create trivy config
cat > workspace/tools/trivy-config.yaml << 'EOF'
# Trivy Configuration
severity:
  - CRITICAL
  - HIGH

vulnerability:
  type:
    - os
    - library

ignore-unfixed: false
EOF

print_success "Trivy configuration created"

# Create SBOM generation script
cat > workspace/tools/generate-sbom.sh << 'EOF'
#!/bin/bash
# SBOM Generation Script

OUTPUT_DIR="workspace/artifacts/sbom"
SOURCE_DIR="."

echo "Generating SBOM..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate SPDX SBOM
echo "Generating SPDX SBOM..."
syft "$SOURCE_DIR" -o spdx-json > "$OUTPUT_DIR/sbom-spdx.json"

# Generate CycloneDX SBOM
echo "Generating CycloneDX SBOM..."
syft "$SOURCE_DIR" -o cyclonedx-json > "$OUTPUT_DIR/sbom-cyclonedx.json"

# Generate SPDX Text format
echo "Generating SPDX Text..."
syft "$SOURCE_DIR" -o spdx-tag-value > "$OUTPUT_DIR/sbom-spdx.txt"

echo "✅ SBOM generation complete"
echo "   SPDX: $OUTPUT_DIR/sbom-spdx.json"
echo "   CycloneDX: $OUTPUT_DIR/sbom-cyclonedx.json"
EOF

chmod +x workspace/tools/generate-sbom.sh
print_success "SBOM generation script created"

# Create vulnerability scanning script
cat > workspace/tools/scan-vulnerabilities.sh << 'EOF'
#!/bin/bash
# Vulnerability Scanning Script

SBOM_FILE="workspace/artifacts/sbom/sbom-spdx.json"
OUTPUT_DIR="workspace/artifacts/security"

echo "Scanning for vulnerabilities..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if SBOM exists
if [ ! -f "$SBOM_FILE" ]; then
    echo "Error: SBOM not found at $SBOM_FILE"
    echo "Please run generate-sbom.sh first"
    exit 1
fi

# Scan SBOM
echo "Scanning SBOM with Trivy..."
trivy sbom \
    --format json \
    --output "$OUTPUT_DIR/vulnerability-scan.json" \
    "$SBOM_FILE"

# Generate human-readable report
echo "Generating vulnerability report..."
trivy sbom \
    --format table \
    --output "$OUTPUT_DIR/vulnerability-report.txt" \
    "$SBOM_FILE"

echo "✅ Vulnerability scan complete"
echo "   JSON: $OUTPUT_DIR/vulnerability-scan.json"
echo "   Report: $OUTPUT_DIR/vulnerability-report.txt"
EOF

chmod +x workspace/tools/scan-vulnerabilities.sh
print_success "Vulnerability scanning script created"

# Create artifact signing script
cat > workspace/tools/sign-artifacts.sh << 'EOF'
#!/bin/bash
# Artifact Signing Script

ARTIFACTS_DIR="workspace/artifacts"
SIGNATURES_DIR="$ARTIFACTS_DIR/signatures"

echo "Signing artifacts..."

# Create output directory
mkdir -p "$SIGNATURES_DIR"

# Sign SBOM
if [ -f "$ARTIFACTS_DIR/sbom/sbom-spdx.json" ]; then
    echo "Signing SBOM..."
    cosign sign-blob \
        --yes \
        --output-certificate "$SIGNATURES_DIR/sbom-cert.pem" \
        --output-signature "$SIGNATURES_DIR/sbom-sig.sig" \
        "$ARTIFACTS_DIR/sbom/sbom-spdx.json"
    echo "✅ SBOM signed"
fi

# Sign provenance
if [ -f "$ARTIFACTS_DIR/provenance/provenance.json" ]; then
    echo "Signing provenance..."
    cosign sign-blob \
        --yes \
        --output-certificate "$SIGNATURES_DIR/provenance-cert.pem" \
        --output-signature "$SIGNATURES_DIR/provenance-sig.sig" \
        "$ARTIFACTS_DIR/provenance/provenance.json"
    echo "✅ Provenance signed"
fi

echo "✅ Artifact signing complete"
echo "   Signatures: $SIGNATURES_DIR"
EOF

chmod +x workspace/tools/sign-artifacts.sh
print_success "Artifact signing script created"

# Create comprehensive test script
cat > workspace/tools/test-supply-chain.sh << 'EOF'
#!/bin/bash
# Supply Chain Testing Script

echo "=== Supply Chain Tools Test ==="
echo ""

# Test syft
echo "Testing syft..."
if syft --version &> /dev/null; then
    echo "✓ syft is working"
    syft --version
else
    echo "✗ syft is not working"
fi

echo ""

# Test trivy
echo "Testing trivy..."
if trivy --version &> /dev/null; then
    echo "✓ trivy is working"
    trivy --version
else
    echo "✗ trivy is not working"
fi

echo ""

# Test cosign
echo "Testing cosign..."
if cosign version &> /dev/null; then
    echo "✓ cosign is working"
    cosign version
else
    echo "✗ cosign is not working"
fi

echo ""

# Test OPA
echo "Testing OPA..."
if opa version &> /dev/null; then
    echo "✓ OPA is working"
    opa version
else
    echo "✗ OPA is not working"
fi

echo ""
echo "=== Test Complete ==="
EOF

chmod +x workspace/tools/test-supply-chain.sh
print_success "Test script created"

# Print summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Installed tools:"
echo "  • syft - SBOM generation"
echo "  • trivy - Vulnerability scanning"
echo "  • cosign - Artifact signing"
echo "  • opa - Policy engine"
echo ""
echo "Created scripts:"
echo "  • workspace/tools/generate-sbom.sh - Generate SBOM"
echo "  • workspace/tools/scan-vulnerabilities.sh - Scan vulnerabilities"
echo "  • workspace/tools/sign-artifacts.sh - Sign artifacts"
echo "  • workspace/tools/test-supply-chain.sh - Test all tools"
echo ""
echo "Configuration files:"
echo "  • workspace/tools/cosign-config.yaml - Cosign configuration"
echo "  • workspace/tools/trivy-config.yaml - Trivy configuration"
echo ""
echo "Next steps:"
echo "  1. Run 'workspace/tools/test-supply-chain.sh' to verify installation"
echo "  2. Run 'workspace/tools/generate-sbom.sh' to generate SBOM"
echo "  3. Run 'workspace/tools/scan-vulnerabilities.sh' to scan for vulnerabilities"
echo "  4. Run 'workspace/tools/sign-artifacts.sh' to sign artifacts"
echo ""
print_success "Supply chain tools setup complete"