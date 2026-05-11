REPO_ROOT := $(shell pwd)
TFLITE_DIR := $(REPO_ROOT)/_third_party/tflite

UNAME_S := $(shell uname -s)
IS_WINDOWS := $(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S))

ifneq ($(IS_WINDOWS),)
# ---- Windows (MSYS2 MINGW64) ----------------------------------------------
# gocv's default cgo directives hardcode C:/opencv/... for Windows, so we
# pass `-tags customenv` and supply OpenCV paths via pkg-config instead.
# go-tflite needs the TFLite headers on the C preprocessor include path.
EXE        := handtracking.exe
BUILD_TAGS := -tags customenv

CV_CFLAGS  := $(shell pkg-config --cflags opencv4)
CV_LDFLAGS := $(shell pkg-config --libs opencv4)

export CGO_CPPFLAGS := -I$(TFLITE_DIR)/include $(CV_CFLAGS)
export CGO_CXXFLAGS := --std=c++11
export CGO_LDFLAGS  := -L$(TFLITE_DIR)/lib -ltensorflowlite_c $(CV_LDFLAGS)
else
# ---- macOS / Linux ---------------------------------------------------------
EXE        := handtracking
BUILD_TAGS :=

export CGO_CFLAGS  := -I$(TFLITE_DIR)/include
export CGO_LDFLAGS := -L$(TFLITE_DIR)/lib -ltensorflowlite_c -Wl,-rpath,$(TFLITE_DIR)/lib
endif

.PHONY: build run test clean tflite

build:
	go build $(BUILD_TAGS) -o $(EXE) .
ifneq ($(IS_WINDOWS),)
	@cp -u $(TFLITE_DIR)/lib/libtensorflowlite_c.dll . 2>/dev/null || cp $(TFLITE_DIR)/lib/libtensorflowlite_c.dll .
endif

run: build
	./$(EXE)

test:
	go test ./internal/...

clean:
	rm -f handtracking handtracking.exe libtensorflowlite_c.dll

# One-time TFLite C library build (~10–15 min). Re-run after upgrading
# the TF version pin in scripts/install-tflite.sh.
tflite:
	./scripts/install-tflite.sh
