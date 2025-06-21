# Nebuchadnezzar - Biological Quantum Computer Framework
# Makefile for development and build automation

.PHONY: all build test check clean doc bench fmt clippy install dev-deps help
.DEFAULT_GOAL := help

# Variables
CARGO := cargo
RUSTC_VERSION := $(shell rustc --version)
PROJECT_NAME := nebuchadnezzar

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Display this help message
	@echo "$(BLUE)Nebuchadnezzar - Biological Quantum Computer Framework$(NC)"
	@echo "$(YELLOW)Rust version: $(RUSTC_VERSION)$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
all: check test build doc ## Run all checks, tests, build and generate docs

build: ## Build the project in release mode
	@echo "$(GREEN)Building project...$(NC)"
	$(CARGO) build --release

build-dev: ## Build the project in debug mode
	@echo "$(GREEN)Building project (debug)...$(NC)"
	$(CARGO) build

# Testing targets
test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(CARGO) test --all-features

test-verbose: ## Run tests with verbose output
	@echo "$(GREEN)Running tests (verbose)...$(NC)"
	$(CARGO) test --all-features -- --nocapture

test-doc: ## Run documentation tests
	@echo "$(GREEN)Running documentation tests...$(NC)"
	$(CARGO) test --doc

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(CARGO) test --test '*'

# Code quality targets
check: ## Run cargo check
	@echo "$(GREEN)Checking code...$(NC)"
	$(CARGO) check --all-features

clippy: ## Run clippy linter
	@echo "$(GREEN)Running clippy...$(NC)"
	$(CARGO) clippy --all-features --all-targets -- -D warnings

clippy-fix: ## Run clippy with automatic fixes
	@echo "$(GREEN)Running clippy with fixes...$(NC)"
	$(CARGO) clippy --all-features --all-targets --fix

fmt: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	$(CARGO) fmt

fmt-check: ## Check code formatting
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(CARGO) fmt -- --check

# Documentation targets
doc: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	$(CARGO) doc --all-features --no-deps

doc-open: ## Generate and open documentation
	@echo "$(GREEN)Generating and opening documentation...$(NC)"
	$(CARGO) doc --all-features --no-deps --open

doc-private: ## Generate documentation including private items
	@echo "$(GREEN)Generating documentation (including private)...$(NC)"
	$(CARGO) doc --all-features --no-deps --document-private-items

# Benchmarking targets
bench: ## Run benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(CARGO) bench

bench-baseline: ## Run benchmarks and save as baseline
	@echo "$(GREEN)Running benchmarks (baseline)...$(NC)"
	$(CARGO) bench -- --save-baseline main

# Example targets
examples: ## Build all examples
	@echo "$(GREEN)Building examples...$(NC)"
	$(CARGO) build --examples

run-glycolysis: ## Run glycolysis circuit example
	@echo "$(GREEN)Running glycolysis circuit example...$(NC)"
	$(CARGO) run --example glycolysis_circuit

run-complete-demo: ## Run complete ATP-oscillatory-membrane demo
	@echo "$(GREEN)Running complete demo...$(NC)"
	$(CARGO) run --example atp_oscillatory_membrane_complete_demo

run-comprehensive: ## Run comprehensive simulation
	@echo "$(GREEN)Running comprehensive simulation...$(NC)"
	$(CARGO) run --example comprehensive_simulation

run-quantum-demo: ## Run quantum biological computer demo
	@echo "$(GREEN)Running quantum biological computer demo...$(NC)"
	$(CARGO) run --example quantum_biological_computer_demo

# Development targets
dev-deps: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	rustup component add clippy rustfmt
	$(CARGO) install cargo-watch cargo-tarpaulin cargo-audit cargo-outdated

watch: ## Watch for changes and run tests
	@echo "$(GREEN)Watching for changes...$(NC)"
	cargo-watch -x check -x test

watch-run: ## Watch for changes and run main example
	@echo "$(GREEN)Watching and running complete demo...$(NC)"
	cargo-watch -x 'run --example atp_oscillatory_membrane_complete_demo'

# Security and maintenance targets
audit: ## Run security audit
	@echo "$(GREEN)Running security audit...$(NC)"
	$(CARGO) audit

outdated: ## Check for outdated dependencies
	@echo "$(GREEN)Checking for outdated dependencies...$(NC)"
	$(CARGO) outdated

update: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(CARGO) update

# Coverage targets
coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	$(CARGO) tarpaulin --all-features --out Html --output-dir coverage

coverage-xml: ## Generate test coverage in XML format
	@echo "$(GREEN)Generating coverage report (XML)...$(NC)"
	$(CARGO) tarpaulin --all-features --out Xml

# Installation targets
install: ## Install the binary
	@echo "$(GREEN)Installing binary...$(NC)"
	$(CARGO) install --path .

uninstall: ## Uninstall the binary
	@echo "$(GREEN)Uninstalling binary...$(NC)"
	$(CARGO) uninstall $(PROJECT_NAME)

# Cleanup targets
clean: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	$(CARGO) clean

clean-all: clean ## Clean all generated files including documentation
	@echo "$(GREEN)Cleaning all generated files...$(NC)"
	rm -rf target/
	rm -rf coverage/
	rm -f Cargo.lock

# Release targets
release-dry: ## Dry run of release build
	@echo "$(GREEN)Dry run release...$(NC)"
	$(CARGO) publish --dry-run

release-check: check test clippy fmt-check doc ## Run all checks before release
	@echo "$(GREEN)All release checks passed!$(NC)"

# Performance profiling targets
profile: ## Run with profiling (requires perf)
	@echo "$(GREEN)Running with profiling...$(NC)"
	$(CARGO) build --release
	perf record --call-graph=dwarf ./target/release/examples/atp_oscillatory_membrane_complete_demo
	perf report

flamegraph: ## Generate flamegraph (requires cargo-flamegraph)
	@echo "$(GREEN)Generating flamegraph...$(NC)"
	$(CARGO) flamegraph --example atp_oscillatory_membrane_complete_demo

# Docker targets (if Docker support is added later)
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME) .

docker-run: ## Run in Docker container
	@echo "$(GREEN)Running in Docker...$(NC)"
	docker run --rm -it $(PROJECT_NAME)

# CI/CD simulation
ci: check clippy fmt-check test doc ## Simulate CI pipeline
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

# Quick development cycle
quick: fmt clippy test ## Quick development cycle: format, lint, test
	@echo "$(GREEN)Quick development cycle completed!$(NC)"

# Full development cycle
full: clean fmt clippy test doc bench ## Full development cycle
	@echo "$(GREEN)Full development cycle completed!$(NC)" 