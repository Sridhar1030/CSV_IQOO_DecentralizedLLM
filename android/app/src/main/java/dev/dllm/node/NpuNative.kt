package dev.dllm.node

/** Thin binding to libdllm_npu.so, which drives LiteRT's C API on the Hexagon HTP. One handle per
 *  layer .tflite. All buffers are float32; the caller matches its tensors to [nativeInputNames] /
 *  [nativeOutputNames] order. See android/app/src/main/cpp/npu_jni.cpp. */
object NpuNative {
    @Volatile private var loaded = false

    /** Loads the native library once. Throws UnsatisfiedLinkError if the device has no usable NPU
     *  runtime, which the caller turns into a fall back to the CPU engine. */
    fun ensureLoaded() {
        if (loaded) return
        synchronized(this) {
            if (!loaded) { System.loadLibrary("dllm_npu"); loaded = true }
        }
    }

    /** Create the LiteRT environment. [pluginDir] holds the Qualcomm dispatch + compiler plugin and
     *  the QNN libraries (the app's nativeLibraryDir); [cacheDir] persists JIT-compiled contexts so
     *  the ~10s per-layer compile is paid once, ever. Returns 0 on failure. */
    external fun nativeInit(pluginDir: String, cacheDir: String): Long
    /** Compile one layer .tflite for the NPU. Returns a handle, or 0 on failure. */
    external fun nativeCreate(env: Long, tflitePath: String): Long
    external fun nativeInputNames(handle: Long): Array<String>
    external fun nativeOutputNames(handle: Long): Array<String>
    /** Write [inputs] (in input-name order), run on the NPU, return outputs in output-name order. */
    external fun nativeRun(handle: Long, inputs: Array<FloatArray>): Array<FloatArray>?
    external fun nativeClose(handle: Long)
    external fun nativeDestroy(env: Long)
}
