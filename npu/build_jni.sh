#!/bin/bash
# Compile the NPU JNI bridge (android/app/src/main/cpp/npu_jni.cpp) to
# android/app/src/main/jniLibs/arm64-v8a/libdllm_npu.so, linked against the nightly libLiteRt.so
# already staged in jniLibs. Done directly with the NDK clang++ (no CMake/ninja needed), so the
# gradle build only has to package jniLibs.
set -e
NDK=${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/26.3.11579264}
LITERT_SRC=${LITERT_SRC:-/Users/chpatel/projects/CSV_IQOO_DecentralizedLLM/litert-src}
CLANG=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android31-clang++
J=android/app/src/main/jniLibs/arm64-v8a
"$CLANG" --std=c++17 -O2 -fPIC -shared \
  -I "$LITERT_SRC" \
  android/app/src/main/cpp/npu_jni.cpp \
  -o "$J/libdllm_npu.so" \
  -L "$J" -lLiteRt -llog -static-libstdc++
echo "built $J/libdllm_npu.so"
$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-readelf -d "$J/libdllm_npu.so" | grep NEEDED
