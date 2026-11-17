CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror
BUILD_DIR ?= /tmp/atlantic-native-build

.PHONY: all lib test fetch-heads clean

all: lib

lib: $(BUILD_DIR)/libatlantic_host.a

fetch-heads:
	python3 fetch_heads.py

$(BUILD_DIR)/atlantic_host.o: src/atlantic_host.cpp include/atlantic_host.hpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -pthread -c $< -o $@

$(BUILD_DIR)/atlantic_ffi.o: src/atlantic_ffi.cpp include/atlantic_host.hpp include/atlantic_host.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -pthread -c $< -o $@

$(BUILD_DIR)/libatlantic_host.a: $(BUILD_DIR)/atlantic_host.o $(BUILD_DIR)/atlantic_ffi.o
	$(AR) rcs $@ $^

$(BUILD_DIR)/host-test: src/atlantic_host.cpp src/atlantic_ffi.cpp include/atlantic_host.hpp include/atlantic_host.h tests/host_test.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -pthread src/atlantic_host.cpp src/atlantic_ffi.cpp tests/host_test.cpp -o $@

test: $(BUILD_DIR)/host-test
	$(BUILD_DIR)/host-test vendor/krummholz-harness
	python3 verify_vendor.py

clean:
	rm -f $(BUILD_DIR)/host-test $(BUILD_DIR)/atlantic_host.o $(BUILD_DIR)/atlantic_ffi.o $(BUILD_DIR)/libatlantic_host.a
