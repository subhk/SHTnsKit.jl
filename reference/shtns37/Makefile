SHTNS_PREFIX ?= /usr/local
FFTW_PREFIX ?= /usr/local
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS += -I$(SHTNS_PREFIX)/include
LDFLAGS += -L$(SHTNS_PREFIX)/lib -L$(FFTW_PREFIX)/lib
LDLIBS += -lshtns -lfftw3 -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LDFLAGS += -Wl,-rpath,$(SHTNS_PREFIX)/lib -Wl,-rpath,$(FFTW_PREFIX)/lib
POST_LINK = install_name_tool -change libshtns.so $(SHTNS_PREFIX)/lib/libshtns.so $@
else
LDFLAGS += -Wl,-rpath,$(SHTNS_PREFIX)/lib -Wl,-rpath,$(FFTW_PREFIX)/lib
POST_LINK = true
endif

.PHONY: all clean
all: generate

generate: generate.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGENERATOR_SOURCE_PATH='"reference/shtns37/generate.c"' $< $(LDFLAGS) $(LDLIBS) -o $@
	$(POST_LINK)

clean:
	$(RM) generate
