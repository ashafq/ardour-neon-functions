#!/usr/bin/env bash
# Test the program with various frame sizes.

set -euo pipefail

# Frame sizes to test
frame_sizes=(41 64 96 128 192 240 256 441 480 512)

# Fixed parameters
alignment=64
iterations=10000000

# Build the code
make 2>&1 >/dev/null

case "$(uname -s)" in
Linux*)
    # Loop over each frame size and run the binary, peg the process
    # to first CPU core
    for frame in "${frame_sizes[@]}"; do
        taskset -c 0 $PWD/test "$frame" "$alignment" "$iterations"
    done
    ;;
*)
    # Loop over each frame size and run the binary
    for frame in "${frame_sizes[@]}"; do
        $PWD/test "$frame" "$alignment" "$iterations"
    done
    ;;
esac
