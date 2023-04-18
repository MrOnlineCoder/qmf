CC=g++
CFLAGS=-Iinclude -std=c++11
SRC_FILES=src/main.cpp src/executor.cpp

build:
	$(CC) $(CFLAGS) -o qmf $(SRC_FILES)

run: build
	./qmf

cltest: src/cltest.cpp src/kernel.cl
	rm -f cltest
	$(CC) $(CFLAGS) -framework OpenCL -o cltest src/cltest.cpp
	./cltest