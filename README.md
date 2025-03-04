# NEON Optimized DSP Functions for Ardour

This project provides ARM NEON–optimized implementations of several common
digital signal processing (DSP) routines in Ardour. The optimized functions are
designed to work on ARM AArch64 systems (such as the Apple M1) and include
microbenchmarking and unit tests for verification and performance comparison
against default (unoptimized) implementations.

## Results from M3 Pro

![M3 Pro](data/m3-pro.png)


## Results from Raspberry Pi Zero 2

![Rasperry Pi Pico](data/pi-zero-2.png)
