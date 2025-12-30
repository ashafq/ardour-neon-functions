#!/usr/bin/env bash
# Test the program with various frame sizes.

set -euo pipefail

# Frame sizes to test
frame_sizes=(5 25 64 128 240 256 441 480 512 768 1024 2048 8192)

# Fixed parameters
alignment=64
iterations=10000000

make 2>&1 > /dev/null

# Loop over each frame size and run the binary
for frame in "${frame_sizes[@]}"; do
    taskset -c 0 ./test "$frame" "$alignment" "$iterations"
done
