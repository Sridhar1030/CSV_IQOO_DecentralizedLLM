package dev.dllm.node

import android.util.Log
import java.io.File
import kotlin.math.cos
import kotlin.math.sin

/** Runs this node's layers on the Qualcomm Hexagon HTP through LiteRT, behind the same [Engine]
 *  interface the pure-Kotlin [CpuEngine] uses, so the service, protocol and UI are unchanged.
 *
 *  Each layer is its own .tflite (hidden in, hidden out, KV cache as explicit tensors), fetched from
 *  the hub as layer_XX.tflite the way the CPU node fetches layer_XX.npz. A node holding layers [a, b)
 *  loads b-a of them and chains them: layer i's hidden_out is layer i+1's hidden_in. The KV cache and
 *  the position-dependent tensors (RoPE cos/sin, the causal mask, the one-hot cache-write, the keep
 *  mask) are computed here on the CPU and passed in, exactly as npu/export_shard.py's host side does,
 *  because the HTP graph must be static: no dynamic shapes, no integer index arithmetic.
 *
 *  The arithmetic matches dllm/np_node.py to fp16: the HTP computes in fp16, and the residual stream
 *  reaches ~1700 on this model, so expect ~1e-2 relative error, invisible after sampling.
 *
 *  Weights are baked into each .tflite as constants, so there is no npz fingerprint to report; the
 *  node sends an empty fingerprint map and the hub shows the layers as present-but-unverified. */
class NpuEngine(private val pluginDir: String, private val cacheDir: String) : Engine {
    override val name = "npu-htp"
    override val shardExt = "tflite"

    private var env = 0L
    private lateinit var cfg: ModelConfig
    private var a = 0
    private val handles = ArrayList<Long>()
    private val inNames = ArrayList<Array<String>>()
    private val outNames = ArrayList<Array<String>>()
    private var cacheLen = 512
    // req -> per local layer [k, v], carried between steps as opaque float32 the graph produced.
    private val cache = HashMap<String, Array<FloatArray>>()

    override fun load(dir: File, a: Int, b: Int, cfg: ModelConfig) {
        this.cfg = cfg; this.a = a
        NpuNative.ensureLoaded()
        if (env == 0L) {
            env = NpuNative.nativeInit(pluginDir, cacheDir)
            require(env != 0L) { "LiteRT NPU environment failed to initialise" }
        }
        for (i in a until b) {
            val path = File(dir, "layer_%02d.tflite".format(i)).absolutePath
            val h = NpuNative.nativeCreate(env, path)
            require(h != 0L) { "NPU compile failed for $path" }
            handles.add(h)
            inNames.add(NpuNative.nativeInputNames(h))
            outNames.add(NpuNative.nativeOutputNames(h))
        }
        // Cache length is whatever the graph was built with: kv_cache_k_0 is [1, kvHeads, headDim, S].
        val kIdx = inNames[0].indexOf("kv_cache_k_0")
        if (kIdx >= 0) cacheLen = kSizeToCacheLen()
        Log.i("dllm", "NpuEngine loaded layers $a-${b - 1}, cache $cacheLen")
    }

    private fun kSizeToCacheLen(): Int = cacheLen  // fixed by the exporter (512); kept for clarity

    private fun kv(req: String): Array<FloatArray> = cache.getOrPut(req) {
        val kElems = cfg.kvHeads * cfg.headDim * cacheLen
        Array(handles.size * 2) { FloatArray(kElems) }   // [2*li]=k, [2*li+1]=v, same element count
    }

    override fun reset(req: String) { cache.remove(req) }

    override fun close() {
        handles.forEach { NpuNative.nativeClose(it) }
        handles.clear()
        if (env != 0L) { NpuNative.nativeDestroy(env); env = 0L }
    }

    // ---- host-side tensors, identical to npu/export_shard.py host_inputs, for one token at `pos` ----

    private fun ropeTables(pos: Int): Pair<FloatArray, FloatArray> {
        val half = cfg.headDim / 2
        val c = FloatArray(half); val s = FloatArray(half)
        for (j in 0 until half) {
            val inv = Math.pow(cfg.theta, -(2.0 * j) / cfg.headDim)
            val f = pos * inv
            c[j] = cos(f).toFloat(); s[j] = sin(f).toFloat()
        }
        return c to s
    }

    private fun causalMask(pos: Int): FloatArray {
        val g = cfg.heads / cfg.kvHeads
        val row = FloatArray(cacheLen) { if (it <= pos) 0f else MASK_FILL }
        val out = FloatArray(g * cacheLen)
        for (r in 0 until g) System.arraycopy(row, 0, out, r * cacheLen, cacheLen)
        return out
    }

    private fun writeMatrix(pos: Int): FloatArray {
        val out = FloatArray(cfg.kvHeads * cacheLen)
        for (h in 0 until cfg.kvHeads) out[h * cacheLen + pos] = 1f
        return out
    }

    private fun keepMask(pos: Int): FloatArray = FloatArray(cacheLen) { if (it == pos) 0f else 1f }

    /** One token through every local layer, extending the KV cache in place. */
    private fun decodeStep(row: FloatArray, pos: Int, req: String): FloatArray {
        val kvs = kv(req)
        val (cosT, sinT) = ropeTables(pos)
        val mask = causalMask(pos); val write = writeMatrix(pos); val keep = keepMask(pos)
        var cur = row
        for (li in handles.indices) {
            val values = mapOf(
                "hidden" to cur, "cos" to cosT, "sin" to sinT, "mask" to mask,
                "write" to write, "keep" to keep,
                "kv_cache_k_0" to kvs[li * 2], "kv_cache_v_0" to kvs[li * 2 + 1],
            )
            val ins = inNames[li]
            val ordered = Array(ins.size) { values[ins[it]] ?: error("NPU layer wants input '${ins[it]}' the host does not supply") }
            val out = NpuNative.nativeRun(handles[li], ordered) ?: error("NPU run failed on local layer $li")
            val outs = outNames[li]
            cur = out[outs.indexOf("hidden_out")]
            kvs[li * 2] = out[outs.indexOf("kv_cache_k_out_0")]
            kvs[li * 2 + 1] = out[outs.indexOf("kv_cache_v_out_0")]
        }
        return cur
    }

    /** n rows of hidden at positions pos..pos+n-1. Decode is n=1; prefill (n>1) is the same causal
     *  step repeated per token, since each token's output depends only on positions <= its own and
     *  the cache accumulates as we go, so the concatenated rows equal a batched prefill exactly. */
    override fun forward(x: FloatArray, n: Int, pos: Int, req: String): FloatArray {
        val h = cfg.hidden
        if (n == 1) return decodeStep(x, pos, req)
        val out = FloatArray(n * h)
        for (t in 0 until n) {
            val row = x.copyOfRange(t * h, (t + 1) * h)
            decodeStep(row, pos + t, req).copyInto(out, t * h)
        }
        return out
    }

    private companion object { const val MASK_FILL = -1e4f }
}
