# Multi-stage build for Nebuchadnezzar - Biological Quantum Computer Framework
# Stage 1: Build environment
FROM rust:1.75-slim as builder

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/nebuchadnezzar

# Copy manifest files
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/

# Create dummy source to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release
RUN rm src/main.rs

# Copy source code
COPY src/ src/
COPY examples/ examples/
COPY docs/ docs/

# Build the actual application
RUN cargo build --release --all-features

# Build examples
RUN cargo build --release --examples

# Stage 2: Runtime environment
FROM debian:bookworm-slim

# Install runtime dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    libssl3 \
    libblas3 \
    liblapack3 \
    libopenblas0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 nebuchadnezzar

# Set working directory
WORKDIR /app

# Copy built binaries from builder stage
COPY --from=builder /usr/src/nebuchadnezzar/target/release/examples/* ./examples/

# Copy documentation and assets
COPY --from=builder /usr/src/nebuchadnezzar/docs/ ./docs/
COPY README.md LICENSE ./

# Set ownership
RUN chown -R nebuchadnezzar:nebuchadnezzar /app

# Switch to non-root user
USER nebuchadnezzar

# Set environment variables for optimal performance
ENV RUST_BACKTRACE=1
ENV RUSTFLAGS="-C target-cpu=native"

# Expose port for potential web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ./examples/atp_oscillatory_membrane_complete_demo --version || exit 1

# Default command - run the complete demo
CMD ["./examples/atp_oscillatory_membrane_complete_demo"]

# Labels for metadata
LABEL maintainer="Kundai Farai Sachikonye <kundai.f.sachikonye@gmail.com>"
LABEL description="Nebuchadnezzar - Biological Quantum Computer Framework using ATP as energy currency"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/fullscreen-triangle/nebuchadnezzar"
LABEL org.opencontainers.image.documentation="https://github.com/fullscreen-triangle/nebuchadnezzar/blob/main/README.md"
LABEL org.opencontainers.image.licenses="MIT" 