CFLAGS=$(shell pkg-config --cflags glib-2.0 gthread-2.0) --std=c99 -Wall -Werror -g
LDFLAGS=$(shell pkg-config --libs glib-2.0 gthread-2.0)
OCLFLAGS=-I/usr/local/cuda/include
SOURCES=main.c ocl.c
BIN=copy-reference-benchmark

all: $(BIN)

$(BIN): $(SOURCES)
	$(CC) $(CFLAGS) $(OCLFLAGS) -O3 -o $(BIN) $(SOURCES) $(LDFLAGS) -lOpenCL

clean:
	rm -f $(BIN)
