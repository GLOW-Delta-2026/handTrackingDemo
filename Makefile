REPO_ROOT := $(shell pwd)
TFLITE_DIR := $(REPO_ROOT)/_third_party/tflite

export CGO_CFLAGS  := -I$(TFLITE_DIR)/include
export CGO_LDFLAGS := -L$(TFLITE_DIR)/lib -ltensorflowlite_c -Wl,-rpath,$(TFLITE_DIR)/lib

.PHONY: build run test clean tflite

build:
	go build -o handtracking .

run: build
	./handtracking

test:
	go test ./internal/...

clean:
	rm -f handtracking

# One-time TFLite C library build (~10–15 min). Re-run after upgrading
# the TF version pin in scripts/install-tflite.sh.
tflite:
	./scripts/install-tflite.sh
