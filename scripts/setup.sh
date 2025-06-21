#!/bin/bash

# Nebuchadnezzar - Biological Quantum Computer Framework
# Development Environment Setup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if running on supported OS
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Check if Rust is installed
check_rust() {
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version)
        print_status "Rust is installed: $RUST_VERSION"
    else
        print_error "Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_header "Installing system dependencies..."
    
    case $OS in
        "linux")
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y \
                    pkg-config \
                    libssl-dev \
                    libblas-dev \
                    liblapack-dev \
                    libopenblas-dev \
                    gfortran \
                    build-essential \
                    cmake \
                    git \
                    curl
            elif command -v yum &> /dev/null; then
                sudo yum install -y \
                    pkgconfig \
                    openssl-devel \
                    blas-devel \
                    lapack-devel \
                    openblas-devel \
                    gcc-gfortran \
                    gcc \
                    gcc-c++ \
                    cmake \
                    git \
                    curl
            else
                print_warning "Could not detect package manager. Please install dependencies manually."
            fi
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                brew install \
                    pkg-config \
                    openssl \
                    openblas \
                    lapack \
                    cmake \
                    git
            else
                print_warning "Homebrew not found. Please install Homebrew and run this script again."
            fi
            ;;
        "windows")
            print_warning "Please ensure you have Visual Studio Build Tools installed."
            ;;
    esac
}

# Install Rust components
install_rust_components() {
    print_header "Installing Rust components..."
    
    rustup component add rustfmt clippy rust-src rust-analyzer
    rustup target add x86_64-unknown-linux-gnu x86_64-apple-darwin x86_64-pc-windows-msvc
}

# Install development tools
install_dev_tools() {
    print_header "Installing development tools..."
    
    cargo install \
        cargo-watch \
        cargo-tarpaulin \
        cargo-audit \
        cargo-outdated \
        cargo-criterion \
        cargo-flamegraph \
        cargo-expand \
        cargo-tree \
        cargo-edit
}

# Setup environment files
setup_env_files() {
    print_header "Setting up environment files..."
    
    if [ ! -f .env ]; then
        cp env.example .env
        print_status "Created .env file from template"
    else
        print_warning ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_header "Creating project directories..."
    
    mkdir -p results
    mkdir -p config
    mkdir -p notebooks
    mkdir -p scripts
    mkdir -p grafana/dashboards
    mkdir -p grafana/datasources
    mkdir -p prometheus
    mkdir -p tests/integration
    mkdir -p tests/fixtures
    
    print_status "Created project directories"
}

# Setup Git hooks
setup_git_hooks() {
    print_header "Setting up Git hooks..."
    
    # Pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Format check
cargo fmt -- --check
if [ $? -ne 0 ]; then
    echo "Code formatting issues found. Run 'cargo fmt' to fix."
    exit 1
fi

# Clippy check
cargo clippy --all-features --all-targets -- -D warnings
if [ $? -ne 0 ]; then
    echo "Clippy issues found. Please fix before committing."
    exit 1
fi

# Test check
cargo test --all-features
if [ $? -ne 0 ]; then
    echo "Tests failed. Please fix before committing."
    exit 1
fi

echo "All pre-commit checks passed!"
EOF

    chmod +x .git/hooks/pre-commit
    print_status "Git pre-commit hook installed"
}

# Generate configuration files
generate_configs() {
    print_header "Generating configuration files..."
    
    # Prometheus configuration
    cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nebuchadnezzar'
    static_configs:
      - targets: ['localhost:8080']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

    # Grafana datasource configuration
    cat > grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    print_status "Configuration files generated"
}

# Build the project
build_project() {
    print_header "Building the project..."
    
    cargo build --all-features
    cargo build --examples
    
    print_status "Project built successfully"
}

# Run tests
run_tests() {
    print_header "Running tests..."
    
    cargo test --all-features
    
    print_status "All tests passed"
}

# Main setup function
main() {
    print_header "=== Nebuchadnezzar Development Environment Setup ==="
    
    check_os
    check_rust
    install_system_deps
    install_rust_components
    install_dev_tools
    setup_env_files
    create_directories
    setup_git_hooks
    generate_configs
    build_project
    run_tests
    
    print_header "=== Setup Complete ==="
    print_status "Development environment is ready!"
    print_status "Run 'make help' to see available commands"
    print_status "Run 'cargo run --example atp_oscillatory_membrane_complete_demo' to test the system"
}

# Run main function
main "$@" 