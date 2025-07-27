# Simple Makefile for Linux/macOS
# Usage:
#   make              # builds shared lib
#   make clean

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBNAME = libl2.dylib
    SHARED_FLAGS = -shared -undefined dynamic_lookup
else
    LIBNAME = libl2.so
    SHARED_FLAGS = -shared
endif

CC      ?= cc
CFLAGS  ?= -O3 -march=native -ffast-math -fPIC
LDFLAGS ?= $(SHARED_FLAGS)

all: $(LIBNAME)

$(LIBNAME): l2.c
	$(CC) $(CFLAGS) l2.c -o $(LIBNAME) $(LDFLAGS)

clean:
	rm -f $(LIBNAME)