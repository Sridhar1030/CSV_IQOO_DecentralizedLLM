package dev.dllm.node

import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

class ModelConfig(j: JSONObject) {
    val hidden = j.getInt("hidden_size")
    val heads = j.getInt("num_attention_heads")
    val kvHeads = j.getInt("num_key_value_heads")
    val headDim = hidden / heads
    val inter = j.getInt("intermediate_size")
    val eps = j.getDouble("rms_norm_eps").toFloat()
    val theta = j.getDouble("rope_theta")
    val nLayers = j.getInt("num_hidden_layers")
}

/** What a runtime has to provide. The service, the protocol and the UI only ever see this, so
 *  swapping the CPU engine for ONNX Runtime, ExecuTorch or llama.cpp is one class and one line
 *  in [Engines]. A node holds layers [a, b) and its own KV cache, keyed by request id. */
interface Engine {
    val name: String
    /** File extension of the per-layer shards this engine downloads: "npz" for the Kotlin CPU
     *  engine, "tflite" for the NPU engine. The hub serves layer_XX.<ext> either way. */
    val shardExt: String get() = "npz"
    /** What one layer really costs this runtime in bytes, when the engine knows better than the
     *  shard size does, else null and the hub sizes us by the shard. The NPU engine reports it: its
     *  graphs carry fp16 weights plus an HTP context, so a 14B layer costs about 2.4 GB against a
     *  131 MB int4 shard, and a planner sizing by the shard would hand a phone far more than it holds. */
    val bytesPerLayer: Long? get() = null
    fun load(dir: File, a: Int, b: Int, cfg: ModelConfig)
    /** x is n rows of cfg.hidden floats at absolute positions pos..pos+n-1. Returns the same shape. */
    fun forward(x: FloatArray, n: Int, pos: Int, req: String): FloatArray
    /** One decode step for several independent requests: x is B rows of cfg.hidden floats, row i
     *  belonging to reqs[i] at positions[i]. Returns B rows. Engines that cannot batch may leave the
     *  default, which simply loops, and the hub will still be correct, only slower. */
    fun forwardBatch(x: FloatArray, positions: IntArray, reqs: Array<String>, hidden: Int): FloatArray {
        val out = FloatArray(x.size)
        for (i in positions.indices) {
            val row = x.copyOfRange(i * hidden, (i + 1) * hidden)
            forward(row, 1, positions[i], reqs[i]).copyInto(out, i * hidden)
        }
        return out
    }
    fun reset(req: String)
    fun close()
}

object Engines {
    /** The one place to change the runtime. */
    fun default(): Engine = CpuEngine()
    fun byName(name: String): Engine = when (name) {
        "cpu" -> CpuEngine()
        else -> throw IllegalArgumentException("no engine called $name")
    }
}

/** Pure Kotlin, float32, every core. The same arithmetic as dllm/np_node.py, so its output
 *  matches the numpy and torch nodes to rounding. Weights stay memory-mapped in native memory
 *  and each matrix row is pulled into a thread-local array once per use, which keeps the hot
 *  loop on plain FloatArrays. This is the reference runtime, not the fast one: it moves weights
 *  as fp32 and decode is bound by that traffic. A quantised engine is the upgrade path. */
class CpuEngine : Engine {
    override val name = "cpu-fp32"
    private lateinit var cfg: ModelConfig
    private lateinit var layers: List<LayerW>
    private val cache = HashMap<String, Array<KV>>()      // req -> per layer
    private val threads = Runtime.getRuntime().availableProcessors()
    private var pool: ExecutorService = Executors.newFixedThreadPool(threads)
    private var loaded: List<Map<String, Tensor>> = emptyList()

    private class LayerW(w: Map<String, Tensor>) {
        val ln1 = w.getValue("input_layernorm.weight").data
        val ln2 = w.getValue("post_attention_layernorm.weight").data
        val wq = w.getValue("self_attn.q_proj.weight"); val bq = w.getValue("self_attn.q_proj.bias").data
        val wk = w.getValue("self_attn.k_proj.weight"); val bk = w.getValue("self_attn.k_proj.bias").data
        val wv = w.getValue("self_attn.v_proj.weight"); val bv = w.getValue("self_attn.v_proj.bias").data
        val wo = w.getValue("self_attn.o_proj.weight")
        val wg = w.getValue("mlp.gate_proj.weight")
        val wu = w.getValue("mlp.up_proj.weight")
        val wd = w.getValue("mlp.down_proj.weight")
    }

    /** Per layer, per request: keys and values for every position seen, kvHeads*headDim each. */
    private class KV { val k = ArrayList<FloatArray>(); val v = ArrayList<FloatArray>() }

    override fun load(dir: File, a: Int, b: Int, cfg: ModelConfig) {
        this.cfg = cfg
        // A reassign must never serve a KV entry computed by the old layers, and the old shard's
        // pages have to go before the new ones come in or two shards sit resident at once. The
        // file handles closed at map time; the mappings themselves only go when the GC collects
        // the buffers, which is the one way Android offers, so ask for a collection here.
        cache.clear(); layers = emptyList(); loaded = emptyList(); System.gc()
        loaded = (a until b).map { Npz.load(File(dir, "layer_%02d.npz".format(it))) }
        layers = loaded.map { LayerW(it) }
    }

    fun fingerprints(a: Int): Map<Int, String> =
        loaded.mapIndexed { i, t -> (a + i) to Npz.fingerprint(t) }.toMap()

    override fun reset(req: String) { cache.remove(req) }
    override fun close() { pool.shutdownNow(); cache.clear() }

    // --------------------------------------------------------------------------------------
    /** out[t, o] = bias[o] + sum_i W[o, i] * x[t, i]. W is (rows, cols) row-major, as PyTorch
     *  stores a Linear. Rows are split across threads; each row is copied out of the mapped
     *  buffer once and reused for all n tokens, so weight traffic is paid once per forward. */
    private fun matmul(w: Tensor, bias: FloatBuffer?, x: FloatArray, n: Int, out: FloatArray, outStride: Int, outOff: Int) {
        val rows = w.rows; val cols = w.cols
        val chunk = (rows + threads - 1) / threads
        val tasks = (0 until rows step chunk).map { r0 ->
            Callable {
                // One reader per task: it decodes an int8 or int4 row into floats on the way in,
                // so everything below is the same dot product whatever the shard stores.
                val src = RowReader(w)
                val row = FloatArray(cols)
                val r1 = minOf(rows, r0 + chunk)
                for (o in r0 until r1) {
                    src.read(o, row)
                    val b = bias?.get(o) ?: 0f
                    for (t in 0 until n) {
                        // Four independent float accumulators. A single Double accumulator forces
                        // every float product to widen and creates one serial dependency chain, so
                        // the JIT can neither vectorise nor overlap the multiplies. Four chains of
                        // floats let it use NEON and keep the pipeline full. Float accumulation also
                        // matches what numpy and torch do, so the answers agree more closely, not less.
                        val xo = t * cols
                        var a0 = 0f; var a1 = 0f; var a2 = 0f; var a3 = 0f
                        var i = 0
                        val lim = cols - 3
                        while (i < lim) {
                            a0 += row[i] * x[xo + i]
                            a1 += row[i + 1] * x[xo + i + 1]
                            a2 += row[i + 2] * x[xo + i + 2]
                            a3 += row[i + 3] * x[xo + i + 3]
                            i += 4
                        }
                        var acc = a0 + a1 + a2 + a3
                        while (i < cols) { acc += row[i] * x[xo + i]; i++ }
                        out[t * outStride + outOff + o] = acc + b
                    }
                }
            }
        }
        pool.invokeAll(tasks).forEach { it.get() }
    }

    private fun rmsNorm(x: FloatArray, n: Int, w: FloatBuffer, out: FloatArray) {
        val h = cfg.hidden
        for (t in 0 until n) {
            var ss = 0.0
            for (i in 0 until h) { val v = x[t * h + i]; ss += v * v }
            val inv = (1.0 / sqrt(ss / h + cfg.eps)).toFloat()
            for (i in 0 until h) out[t * h + i] = x[t * h + i] * inv * w.get(i)
        }
    }

    /** Rotary embedding, rotate-half form, in place on one head vector. */
    private fun rope(v: FloatArray, off: Int, pos: Int) {
        val d = cfg.headDim; val half = d / 2
        for (i in 0 until half) {
            val inv = 1.0 / cfg.theta.pow(2.0 * i / d)
            val f = pos * inv
            val c = cos(f).toFloat(); val s = sin(f).toFloat()
            val x1 = v[off + i]; val x2 = v[off + half + i]
            v[off + i] = x1 * c - x2 * s
            v[off + half + i] = x2 * c + x1 * s
        }
    }

    /** The batched decode step. Every projection and the whole feedforward run over all B rows in
     *  one matmul, so each weight row is pulled out of the mapped file once for the entire batch.
     *  That is where the win is: a single decode step reads far more weight bytes than it does
     *  arithmetic, so paying that cost once for B requests is nearly free. Attention is done per
     *  row, since each request has its own history and its own position. */
    override fun forwardBatch(x0: FloatArray, positions: IntArray, reqs: Array<String>, hidden: Int): FloatArray {
        val h = cfg.hidden; val hd = cfg.headDim; val H = cfg.heads; val KVH = cfg.kvHeads
        val rep = H / KVH
        val B = positions.size
        val kvs = reqs.map { r -> cache.getOrPut(r) { Array(layers.size) { KV() } } }
        var x = x0.copyOf()
        val hn = FloatArray(B * h)
        val q = FloatArray(B * H * hd); val k = FloatArray(B * KVH * hd); val v = FloatArray(B * KVH * hd)
        val att = FloatArray(B * h)
        val g = FloatArray(B * cfg.inter); val u = FloatArray(B * cfg.inter)
        val proj = FloatArray(B * h)
        val scale = (1.0 / sqrt(hd.toDouble())).toFloat()

        for (li in layers.indices) {
            val L = layers[li]
            rmsNorm(x, B, L.ln1, hn)
            matmul(L.wq, L.bq, hn, B, q, H * hd, 0)
            matmul(L.wk, L.bk, hn, B, k, KVH * hd, 0)
            matmul(L.wv, L.bv, hn, B, v, KVH * hd, 0)
            for (i in 0 until B) {
                val kv = kvs[i][li]
                for (hh in 0 until H) rope(q, i * H * hd + hh * hd, positions[i])
                for (gg in 0 until KVH) rope(k, i * KVH * hd + gg * hd, positions[i])
                kv.k.add(k.copyOfRange(i * KVH * hd, (i + 1) * KVH * hd))
                kv.v.add(v.copyOfRange(i * KVH * hd, (i + 1) * KVH * hd))
                val last = kv.k.size - 1                      // one query attends over all of them
                for (hh in 0 until H) {
                    val gg = hh / rep
                    val qo = i * H * hd + hh * hd
                    val scores = FloatArray(last + 1)
                    var mx = Float.NEGATIVE_INFINITY
                    for (j in 0..last) {
                        val kj = kv.k[j]; var sdot = 0f
                        for (d in 0 until hd) sdot += q[qo + d] * kj[gg * hd + d]
                        sdot *= scale; scores[j] = sdot; if (sdot > mx) mx = sdot
                    }
                    var sum = 0f
                    for (j in 0..last) { val e = exp(scores[j] - mx); scores[j] = e; sum += e }
                    val ao = i * h + hh * hd
                    for (d in 0 until hd) att[ao + d] = 0f
                    for (j in 0..last) {
                        val pw = scores[j] / sum; val vj = kv.v[j]
                        for (d in 0 until hd) att[ao + d] += pw * vj[gg * hd + d]
                    }
                }
            }
            matmul(L.wo, null, att, B, proj, h, 0)
            for (i in x.indices) x[i] += proj[i]
            rmsNorm(x, B, L.ln2, hn)
            matmul(L.wg, null, hn, B, g, cfg.inter, 0)
            matmul(L.wu, null, hn, B, u, cfg.inter, 0)
            for (i in g.indices) { val gi = g[i]; g[i] = gi / (1f + exp(-gi)) * u[i] }
            matmul(L.wd, null, g, B, proj, h, 0)
            for (i in x.indices) x[i] += proj[i]
        }
        return x
    }

    override fun forward(x0: FloatArray, n: Int, pos: Int, req: String): FloatArray {
        val h = cfg.hidden; val hd = cfg.headDim; val H = cfg.heads; val KVH = cfg.kvHeads
        val rep = H / KVH
        val kvs = cache.getOrPut(req) { Array(layers.size) { KV() } }
        var x = x0.copyOf()
        val hn = FloatArray(n * h)
        val q = FloatArray(n * H * hd); val k = FloatArray(n * KVH * hd); val v = FloatArray(n * KVH * hd)
        val att = FloatArray(n * h)
        val g = FloatArray(n * cfg.inter); val u = FloatArray(n * cfg.inter)
        val proj = FloatArray(n * h)
        val scale = (1.0 / sqrt(hd.toDouble())).toFloat()

        for ((li, L) in layers.withIndex()) {
            val kv = kvs[li]
            rmsNorm(x, n, L.ln1, hn)
            matmul(L.wq, L.bq, hn, n, q, H * hd, 0)
            matmul(L.wk, L.bk, hn, n, k, KVH * hd, 0)
            matmul(L.wv, L.bv, hn, n, v, KVH * hd, 0)
            for (t in 0 until n) {
                for (hh in 0 until H) rope(q, t * H * hd + hh * hd, pos + t)
                for (gg in 0 until KVH) rope(k, t * KVH * hd + gg * hd, pos + t)
                kv.k.add(k.copyOfRange(t * KVH * hd, (t + 1) * KVH * hd))
                kv.v.add(v.copyOfRange(t * KVH * hd, (t + 1) * KVH * hd))
            }
            val P = kv.k.size - n                       // positions already cached before this call
            for (t in 0 until n) {
                val last = P + t                        // causal: attend to 0..last inclusive
                for (hh in 0 until H) {
                    val gg = hh / rep
                    val qo = t * H * hd + hh * hd
                    val scores = FloatArray(last + 1)
                    var mx = Float.NEGATIVE_INFINITY
                    for (j in 0..last) {
                        val kj = kv.k[j]; var s = 0f
                        for (d in 0 until hd) s += q[qo + d] * kj[gg * hd + d]
                        s *= scale; scores[j] = s; if (s > mx) mx = s
                    }
                    var sum = 0f
                    for (j in 0..last) { val e = exp(scores[j] - mx); scores[j] = e; sum += e }
                    val ao = t * h + hh * hd
                    for (d in 0 until hd) att[ao + d] = 0f
                    for (j in 0..last) {
                        val p = scores[j] / sum; val vj = kv.v[j]
                        for (d in 0 until hd) att[ao + d] += p * vj[gg * hd + d]
                    }
                }
            }
            matmul(L.wo, null, att, n, proj, h, 0)
            for (i in x.indices) x[i] += proj[i]
            rmsNorm(x, n, L.ln2, hn)
            matmul(L.wg, null, hn, n, g, cfg.inter, 0)
            matmul(L.wu, null, hn, n, u, cfg.inter, 0)
            for (i in g.indices) { val gi = g[i]; g[i] = gi / (1f + exp(-gi)) * u[i] }   // silu(g) * u
            matmul(L.wd, null, g, n, proj, h, 0)
            for (i in x.indices) x[i] += proj[i]
        }
        return x
    }
}
