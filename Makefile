CC=g++
CFLAGS=-Iinclude
SRC_FILES=src/main.cpp src/executor.cpp

build:
	$(CC) $(CFLAGS) -o qmf $(SRC_FILES)

run: build
	./qmf
