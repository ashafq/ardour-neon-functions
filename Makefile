# Source files
SOURCES := arm_neon_functions.cc default_functions.cc test.cc
OBJECTS := $(SOURCES:.cc=.o)
CFLAGS := -std=c++20 -O3 -ffast-math -fomit-frame-pointer -fstrength-reduce -DARM_NEON_SUPPORT=1

# Determine compiler and flags based on OS/Arch
ifneq ($(shell uname -a | grep -c "Linux.*armv7l"), 0)
    CXX := /usr/bin/g++
    CFLAGS += -fopenmp -march=armv7-a -mfpu=neon -mfloat-abi=hard
    LDFLAGS := -lm
else ifneq ($(shell uname -a | grep -c "Linux.*aarch64"), 0)
    CXX := /usr/bin/g++
    CFLAGS += -fopenmp -march=armv8-a+simd
    LDFLAGS := -lm
else ifneq ($(shell uname -a | grep -c "Darwin.*arm64"), 0)
    CXX := /usr/bin/clang++
    CFLAGS += -march=armv8-a+simd
    LDFLAGS := -framework accelerate
else
    $(error "Unsupported target: $(shell uname -a)")
endif

# Default target
all: test

# Compile object files
arm_neon_functions.o: arm_neon_functions.cc
	$(CXX) $(CFLAGS) -c $< -o $@

default_functions.o: default_functions.cc
	$(CXX) $(CFLAGS) -c $< -o $@

test.o: test.cc
	$(CXX) $(CFLAGS) -c $< -o $@

# Link the test executable
test: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Clean up generated files
clean:
	rm -f *.o test

# Format all .cc files
format:
	clang-format -i ./*.cc
