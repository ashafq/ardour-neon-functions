#!/usr/bin/env bash
# Test the program with various frame sizes.

set -euo pipefail

# Frame sizes to test
frame_sizes=(128 240 256 441 480 512 768 1024 2048 8192)

# Fixed parameters
alignment=16
iterations=10000000

make

# Loop over each frame size and run the binary
for frame in "${frame_sizes[@]}"; do
    echo -n "Frame Size: $frame "
    ./test "$frame" "$alignment" "$iterations"
done
