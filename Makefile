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

# Locate OpenCV. Prefer pkg-config (canonical); fall back to MSYS2's default
# install prefix /mingw64 if pkg-config can't find the .pc file. If neither
# works, abort with a clear message rather than letting gocv fail with a
# confusing "opencv2/opencv.hpp: No such file or directory" downstream.
CV_CFLAGS  := $(shell pkg-config --cflags opencv4 2>/dev/null)
CV_LDFLAGS := $(shell pkg-config --libs opencv4 2>/dev/null)

ifeq ($(strip $(CV_CFLAGS)),)
  ifneq (,$(wildcard /mingw64/include/opencv4/opencv2/opencv.hpp))
    CV_CFLAGS  := -I/mingw64/include/opencv4
    CV_LDFLAGS := -L/mingw64/bin -L/mingw64/lib \
      -lopencv_dnn -lopencv_video -lopencv_photo -lopencv_objdetect \
      -lopencv_calib3d -lopencv_features2d -lopencv_videoio -lopencv_imgcodecs \
      -lopencv_imgproc -lopencv_highgui -lopencv_core
    $(warning pkg-config could not find opencv4; using MSYS2 fallback paths under /mingw64)
  else
    $(error OpenCV not found. Install it inside MSYS2 MINGW64 with: pacman -S --needed mingw-w64-x86_64-opencv mingw-w64-x86_64-pkgconf)
  endif
endif

# cgo splits include flags by compiler:
#   CGO_CFLAGS    -> the C compiler (used for go-tflite's C glue)
#   CGO_CXXFLAGS  -> the C++ compiler (used for gocv's .cpp wrappers)
# Putting -I in CPPFLAGS alone is NOT enough: gocv's C++ files won't see it
# and fail with "opencv2/opencv.hpp: No such file or directory".
export CGO_CFLAGS   := -I$(TFLITE_DIR)/include
export CGO_CXXFLAGS := --std=c++11 $(CV_CFLAGS)
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
