# Source files
SOURCES := arm_neon_functions.cc x86_functions_avx512f.cc default_functions.cc test.cc
OBJECTS := $(SOURCES:.cc=.o)
CFLAGS := -std=c++20 -O3 -ffast-math -fomit-frame-pointer -fstrength-reduce

# Determine compiler and flags based on OS/Arch
ifneq ($(shell uname -a | grep -c "Linux.*armv7l"), 0)
    CXX := /usr/bin/g++
    CFLAGS += -fopenmp -march=armv7-a -mfpu=neon -mfloat-abi=hard -DARM_NEON_SUPPORT=1
    LDFLAGS := -lm
else ifneq ($(shell uname -a | grep -c "Linux.*aarch64"), 0)
    CXX := /usr/bin/g++
    CFLAGS += -fopenmp -march=armv8-a+simd -DARM_NEON_SUPPORT=1
    LDFLAGS := -lm
else ifneq ($(shell uname -a | grep -c "Darwin.*arm64"), 0)
    CXX := /usr/bin/clang++
    CFLAGS += -march=armv8-a+simd -DARM_NEON_SUPPORT=1
    LDFLAGS := -framework accelerate
# Check for x86_64 target
else ifneq ($(shell uname -a | grep -ic "linux.*x86_64"), 0)
    CXX := /usr/bin/g++
    CFLAGS += -fopenmp
    ifneq ($(shell lscpu | grep -ic "avx512"), 0)
        CFLAGS += -mavx512f -mfma -DFPU_AVX512F_SUPPORT=1
    else
        $(error "Unsupported extension on x86_64")
    endif
    LDFLAGS := -lm
else
    $(error "Unsupported target: $(shell uname -a)")
endif

# Default target
all: test

# Compile object files
arm_neon_functions.o: arm_neon_functions.cc
	$(CXX) $(CFLAGS) -c $< -o $@

x86_functions_avx512f.o: x86_functions_avx512f.cc
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
