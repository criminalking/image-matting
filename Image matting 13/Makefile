OPENCV_LD_FLAGS := $(shell pkg-config --libs opencv)
OPENCV_C_FLAGS := $(shell pkg-config --cflags opencv)

GXX := clang++
BIN_NAME := main
CPP_FILES := $(wildcard ./*.cpp)

build:
	@$(GXX) main.cpp Imagematting.cpp $(OPENCV_LD_FLAGS) $(OPENCV_C_FLAGS) -o $(BIN_NAME)

run:
	@./$(BIN_NAME)
