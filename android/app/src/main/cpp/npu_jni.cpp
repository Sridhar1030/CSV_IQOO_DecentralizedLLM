// JNI bridge from the Android node's NpuEngine to LiteRT's C API, so a phone runs its layers on the
// Qualcomm Hexagon HTP. One CompiledModel per layer .tflite (hidden in, hidden out, KV cache as
// explicit tensors); the Kotlin side holds the cache and computes RoPE/mask/write on the host.
//
// Linked against the nightly libLiteRt.so, NOT the 2.x Maven release: the release runtime SIGBUSes
// on this SoC (Snapdragon 8 Elite Gen 5, Hexagon V81), reproduced with benchmark_model. See
// how_torun_on_npu.md. The Qualcomm dispatch/compiler plugins and QNN 2.49 libraries sit beside this
// .so in the app's nativeLibraryDir, which is passed in as pluginDir/dispatchDir; LiteRT also points
// ADSP_LIBRARY_PATH there so the DSP loads libQnnHtpV81Skel.so.
#include <jni.h>
#include <android/log.h>
#include <cstring>
#include <string>
#include <vector>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

#define LOG_TAG "dllm-npu"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Everything one layer needs, created once at load and reused every decode step.
struct LayerModel {
  LiteRtModel model = nullptr;
  LiteRtOptions options = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  std::vector<std::string> in_names, out_names;
  std::vector<LiteRtTensorBuffer> in_bufs, out_bufs;
  std::vector<size_t> in_floats, out_floats;   // packed element counts, for memcpy and validation
};

size_t Prod(const LiteRtLayout& l) {
  size_t n = 1;
  for (unsigned int i = 0; i < l.rank; ++i) n *= (size_t)l.dimensions[i];
  return n;
}

bool Ok(LiteRtStatus s, const char* what) {
  if (s != kLiteRtStatusOk) { LOGE("%s failed: status %d", what, (int)s); return false; }
  return true;
}

}  // namespace

extern "C" {

// Create the LiteRT environment: where to find the compiler plugin + dispatch libs and the JIT cache.
JNIEXPORT jlong JNICALL
Java_dev_dllm_node_NpuNative_nativeInit(JNIEnv* env, jclass, jstring plugin_dir, jstring cache_dir) {
  const char* plugin = env->GetStringUTFChars(plugin_dir, nullptr);
  const char* cache = env->GetStringUTFChars(cache_dir, nullptr);
  LiteRtEnvOption opts[3];
  opts[0].tag = kLiteRtEnvOptionTagCompilerPluginLibraryDir;
  opts[0].value.type = kLiteRtAnyTypeString; opts[0].value.str_value = plugin;
  opts[1].tag = kLiteRtEnvOptionTagDispatchLibraryDir;
  opts[1].value.type = kLiteRtAnyTypeString; opts[1].value.str_value = plugin;
  opts[2].tag = kLiteRtEnvOptionTagCompilerCacheDir;
  opts[2].value.type = kLiteRtAnyTypeString; opts[2].value.str_value = cache;
  LiteRtEnvironment e = nullptr;
  LiteRtStatus s = LiteRtCreateEnvironment(3, opts, &e);
  LOGI("nativeInit plugin=%s cache=%s -> status %d", plugin, cache, (int)s);
  env->ReleaseStringUTFChars(plugin_dir, plugin);
  env->ReleaseStringUTFChars(cache_dir, cache);
  return Ok(s, "LiteRtCreateEnvironment") ? (jlong)e : 0;
}

// Load one layer .tflite and compile it for the NPU. Buffers are allocated once and kept.
JNIEXPORT jlong JNICALL
Java_dev_dllm_node_NpuNative_nativeCreate(JNIEnv* env, jclass, jlong env_handle, jstring path_j) {
  auto* lenv = (LiteRtEnvironment)env_handle;
  const char* path = env->GetStringUTFChars(path_j, nullptr);
  auto fail = [&](const char* w) -> jlong { LOGE("nativeCreate(%s): %s failed", path, w);
      env->ReleaseStringUTFChars(path_j, path); return 0; };

  auto* m = new LayerModel();
  if (!Ok(LiteRtCreateModelFromFile(lenv, path, &m->model), "CreateModelFromFile")) { delete m; return fail("model"); }
  if (!Ok(LiteRtCreateOptions(&m->options), "CreateOptions")) { delete m; return fail("options"); }
  if (!Ok(LiteRtSetOptionsHardwareAccelerators(m->options, kLiteRtHwAcceleratorNpu), "SetAccelerators")) { delete m; return fail("accel"); }
  if (!Ok(LiteRtCreateCompiledModel(lenv, m->model, m->options, &m->compiled), "CreateCompiledModel")) { delete m; return fail("compile"); }

  LiteRtSignature sig = nullptr;
  if (!Ok(LiteRtGetModelSignature(m->model, 0, &sig), "GetModelSignature")) { delete m; return fail("sig"); }
  LiteRtParamIndex nin = 0, nout = 0;
  LiteRtGetNumSignatureInputs(sig, &nin);
  LiteRtGetNumSignatureOutputs(sig, &nout);

  // Inputs: name, packed element count from the layout, and a managed buffer from the HW requirements.
  for (LiteRtParamIndex i = 0; i < nin; ++i) {
    const char* nm = nullptr; LiteRtGetSignatureInputName(sig, i, &nm);
    LiteRtLayout layout; if (!Ok(LiteRtGetCompiledModelInputTensorLayout(m->compiled, 0, i, &layout), "InLayout")) { delete m; return fail("inlayout"); }
    LiteRtRankedTensorType tt; tt.element_type = kLiteRtElementTypeFloat32; tt.layout = layout;
    LiteRtTensorBufferRequirements req = nullptr;
    if (!Ok(LiteRtGetCompiledModelInputBufferRequirements(m->compiled, 0, i, &req), "InReq")) { delete m; return fail("inreq"); }
    LiteRtTensorBuffer buf = nullptr;
    if (!Ok(LiteRtCreateManagedTensorBufferFromRequirements(lenv, &tt, req, &buf), "InBuf")) { delete m; return fail("inbuf"); }
    m->in_names.emplace_back(nm ? nm : "");
    m->in_floats.push_back(Prod(layout));
    m->in_bufs.push_back(buf);
  }
  // Outputs: all layouts at once, then a buffer each.
  std::vector<LiteRtLayout> olay(nout);
  if (nout && !Ok(LiteRtGetCompiledModelOutputTensorLayouts(m->compiled, 0, nout, olay.data(), false), "OutLayouts")) { delete m; return fail("outlayouts"); }
  for (LiteRtParamIndex i = 0; i < nout; ++i) {
    const char* nm = nullptr; LiteRtGetSignatureOutputName(sig, i, &nm);
    LiteRtRankedTensorType tt; tt.element_type = kLiteRtElementTypeFloat32; tt.layout = olay[i];
    LiteRtTensorBufferRequirements req = nullptr;
    if (!Ok(LiteRtGetCompiledModelOutputBufferRequirements(m->compiled, 0, i, &req), "OutReq")) { delete m; return fail("outreq"); }
    LiteRtTensorBuffer buf = nullptr;
    if (!Ok(LiteRtCreateManagedTensorBufferFromRequirements(lenv, &tt, req, &buf), "OutBuf")) { delete m; return fail("outbuf"); }
    m->out_names.emplace_back(nm ? nm : "");
    m->out_floats.push_back(Prod(olay[i]));
    m->out_bufs.push_back(buf);
  }
  LOGI("nativeCreate(%s): %d inputs, %d outputs, compiled for NPU", path, (int)nin, (int)nout);
  env->ReleaseStringUTFChars(path_j, path);
  return (jlong)m;
}

JNIEXPORT jobjectArray JNICALL
Java_dev_dllm_node_NpuNative_nativeInputNames(JNIEnv* env, jclass, jlong h) {
  auto* m = (LayerModel*)h;
  jobjectArray a = env->NewObjectArray(m->in_names.size(), env->FindClass("java/lang/String"), nullptr);
  for (size_t i = 0; i < m->in_names.size(); ++i) env->SetObjectArrayElement(a, i, env->NewStringUTF(m->in_names[i].c_str()));
  return a;
}

JNIEXPORT jobjectArray JNICALL
Java_dev_dllm_node_NpuNative_nativeOutputNames(JNIEnv* env, jclass, jlong h) {
  auto* m = (LayerModel*)h;
  jobjectArray a = env->NewObjectArray(m->out_names.size(), env->FindClass("java/lang/String"), nullptr);
  for (size_t i = 0; i < m->out_names.size(); ++i) env->SetObjectArrayElement(a, i, env->NewStringUTF(m->out_names[i].c_str()));
  return a;
}

// Write inputs (in input-name order), run signature 0 on the NPU, read outputs (in output-name order).
JNIEXPORT jobjectArray JNICALL
Java_dev_dllm_node_NpuNative_nativeRun(JNIEnv* env, jclass, jlong h, jobjectArray inputs) {
  auto* m = (LayerModel*)h;
  const size_t nin = m->in_bufs.size(), nout = m->out_bufs.size();
  for (size_t i = 0; i < nin; ++i) {
    auto arr = (jfloatArray)env->GetObjectArrayElement(inputs, i);
    jsize len = env->GetArrayLength(arr);
    if ((size_t)len != m->in_floats[i]) { LOGE("input %zu (%s): got %d floats, want %zu", i, m->in_names[i].c_str(), (int)len, m->in_floats[i]); return nullptr; }
    void* dst = nullptr;
    if (!Ok(LiteRtLockTensorBuffer(m->in_bufs[i], &dst, kLiteRtTensorBufferLockModeWrite), "LockIn")) return nullptr;
    env->GetFloatArrayRegion(arr, 0, len, (jfloat*)dst);
    LiteRtUnlockTensorBuffer(m->in_bufs[i]);
    env->DeleteLocalRef(arr);
  }
  if (!Ok(LiteRtRunCompiledModel(m->compiled, 0, nin, m->in_bufs.data(), nout, m->out_bufs.data()), "Run")) return nullptr;
  jobjectArray out = env->NewObjectArray(nout, env->FindClass("[F"), nullptr);
  for (size_t i = 0; i < nout; ++i) {
    void* src = nullptr;
    if (!Ok(LiteRtLockTensorBuffer(m->out_bufs[i], &src, kLiteRtTensorBufferLockModeRead), "LockOut")) return nullptr;
    jfloatArray arr = env->NewFloatArray(m->out_floats[i]);
    env->SetFloatArrayRegion(arr, 0, m->out_floats[i], (const jfloat*)src);
    LiteRtUnlockTensorBuffer(m->out_bufs[i]);
    env->SetObjectArrayElement(out, i, arr);
    env->DeleteLocalRef(arr);
  }
  return out;
}

JNIEXPORT void JNICALL
Java_dev_dllm_node_NpuNative_nativeClose(JNIEnv*, jclass, jlong h) {
  auto* m = (LayerModel*)h;
  if (!m) return;
  for (auto b : m->in_bufs) LiteRtDestroyTensorBuffer(b);
  for (auto b : m->out_bufs) LiteRtDestroyTensorBuffer(b);
  if (m->compiled) LiteRtDestroyCompiledModel(m->compiled);
  if (m->options) LiteRtDestroyOptions(m->options);
  if (m->model) LiteRtDestroyModel(m->model);
  delete m;
}

JNIEXPORT void JNICALL
Java_dev_dllm_node_NpuNative_nativeDestroy(JNIEnv*, jclass, jlong env_handle) {
  if (env_handle) LiteRtDestroyEnvironment((LiteRtEnvironment)env_handle);
}

}  // extern "C"
