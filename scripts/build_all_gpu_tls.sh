#!/usr/bin/env bash
set -euo pipefail

# Build all default llama.cpp executables with CUDA + OpenSSL enabled.
#
# Environment overrides:
#   BUILD_DIR  - CMake build directory (default: build)
#   JOBS       - Parallel build jobs passed as `-j` (default: auto-detect)
#
# Usage:
#   scripts/build_all_gpu_tls.sh
#   BUILD_DIR=build-cuda-tls JOBS=16 scripts/build_all_gpu_tls.sh

usage() {
    cat <<'EOF'
Usage: build_all_gpu_tls.sh [--help]

Configures and builds llama.cpp in Release mode with:
  - CUDA backend enabled   (GGML_CUDA=ON)
  - TLS/OpenSSL enabled    (LLAMA_OPENSSL=ON)

Environment overrides:
  BUILD_DIR   CMake build directory (default: build)
  JOBS        Parallel build jobs for cmake --build -j (default: auto)

Examples:
  build_all_gpu_tls.sh
  BUILD_DIR=build-cuda-tls JOBS=12 build_all_gpu_tls.sh
EOF
}

if [[ ${1-} == "-h" || ${1-} == "--help" ]]; then
    usage
    exit 0
fi

# No positional arguments are expected.
if [[ $# -gt 0 ]]; then
    usage
    exit 1
fi

# Resolve repository root from this script location.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc)}"

echo "[1/2] Configuring: ${BUILD_DIR} (CUDA + OpenSSL, Release)"
cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/${BUILD_DIR}" \
    -DGGML_CUDA=ON \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_BUILD_TYPE=Release

echo "[2/2] Building all default targets with -j ${JOBS}"
cmake --build "${REPO_ROOT}/${BUILD_DIR}" --config Release -j "${JOBS}"

echo "Done. Binaries are under: ${REPO_ROOT}/${BUILD_DIR}/bin"
