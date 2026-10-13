#!/bin/bash
set -e
cd "$(dirname "$0")"
rm -rf build
cmake -B build
cmake --build build -j"$(nproc)"
echo "OK: ./build/example ./build/bench"
