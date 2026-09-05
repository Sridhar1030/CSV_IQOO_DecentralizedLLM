package dev.dllm.node

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** The node protocol, the same one dllm/np_node.py speaks. One persistent WebSocket to the hub.
 *  hello -> assign -> download only the assigned layers -> ready -> answer fwd frames, heartbeat
 *  every second, reclaim the same range if the connection drops. */
class Node(
    private val hub: String,            // host:port
    private val code: String,
    private val nodeName: String,
    private val engine: Engine,
    private val shardDir: File,
    private val stats: Stats,
    private val ui: (status: String, detail: String) -> Unit,
    private val onForward: () -> Unit,
) {
    @Volatile var running = true
    @Volatile private var layers: IntArray? = null
    private var ws: WebSocket? = null
    private var cfg: ModelConfig? = null
    private var forwards = 0
    private val http = OkHttpClient.Builder()
        .pingInterval(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private val httpBase get() = "http://$hub"
    // Forwards run here, never on the socket's reading thread. A long prefill used to block that
    // thread for tens of seconds, so the node could not answer the hub's pings and the connection
    // was dropped mid-request. One thread, so frames are still processed strictly in order.
    private var work = Executors.newSingleThreadExecutor()

    fun stop() { running = false; ws?.close(1000, "leave"); work.shutdownNow(); engine.close() }

    /** Blocks. Reconnects forever until [stop]. */
    fun run() {
        while (running) {
            val done = java.util.concurrent.CountDownLatch(1)
            ui("connecting", hub)
            ws = http.newWebSocket(Request.Builder().url("ws://$hub/ws/node").build(), object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    val hello = JSONObject().put("t", "hello").put("name", nodeName).put("code", code)
                        .put("device", "android-${engine.name}").put("ram_gb", stats.ramGb())
                    layers?.let { hello.put("layers", JSONArray(listOf(it[0], it[1]))) }
                    webSocket.send(Wire.pack(hello).toByteString())
                }
                override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                    val frame = bytes.toByteArray()
                    work.execute {
                        try { handle(webSocket, frame) }
                        catch (e: Exception) { Log.e("dllm", "frame handling failed", e); ui("error", e.toString()); webSocket.close(1011, e.message) }
                    }
                }
                // The hub closing first sends a close frame; OkHttp waits for us to answer it before
                // onClosed fires. Without this, a hub restart left the node waiting forever.
                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    Log.i("dllm", "hub closed the socket: $code $reason"); webSocket.close(1000, null); done.countDown()
                }
                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) { done.countDown() }
                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    Log.i("dllm", "socket failed: $t"); ui("disconnected", t.message ?: t.toString()); done.countDown()
                }
            })
            done.await()
            hbThread?.interrupt()
            // A queued forward from the dead socket must not run against the new one.
            work.shutdownNow(); work = Executors.newSingleThreadExecutor()
            if (running) { ui("retrying in 3s", ""); Log.i("dllm", "reconnecting to $hub"); Thread.sleep(3000) }
        }
    }

    private var hbThread: Thread? = null

    private fun handle(ws: WebSocket, frame: ByteArray) {
        val (hdr, payload) = Wire.unpack(frame)
        when (hdr.getString("t")) {
            "assign" -> {
                val range = hdr.getJSONArray("layers")
                val a = range.getInt(0); val b = range.getInt(1)
                layers = intArrayOf(a, b)
                ui("downloading", "layers $a-${b - 1}")
                shardDir.mkdirs()
                fetch("config.json")
                val ext = engine.shardExt
                for (i in a until b) fetch("layer_%02d.$ext".format(i), "file ${i - a + 1} of ${b - a}  (layers $a-${b - 1})")
                dropForeign(a, b, ext)
                val c = ModelConfig(JSONObject(File(shardDir, "config.json").readText())); cfg = c
                ui("loading", "layers $a-${b - 1} into ${engine.name}")
                val t0 = System.nanoTime()
                engine.load(shardDir, a, b, c)
                val z = FloatArray(c.hidden)
                engine.forward(z, 1, 0, "_b"); engine.reset("_b")
                val t1 = System.nanoTime(); engine.forward(z, 1, 0, "_b"); engine.reset("_b")
                val msPerLayer = (System.nanoTime() - t1) / 1e6 / (b - a)
                ui("hashing", "verifying weights against the manifest")
                val fps = (engine as? CpuEngine)?.fingerprints(a) ?: emptyMap()
                val ready = JSONObject().put("t", "ready").put("layers", JSONArray(listOf(a, b)))
                    .put("ms_per_layer", msPerLayer).put("batch", true)
                    .put("rss_mb", stats.rssBytes() / 1048576.0)
                    .put("shard_dir", shardDir.absolutePath)
                    .put("files", JSONArray(shardDir.list()!!.sorted()))
                    .put("fingerprints", JSONObject(fps.mapKeys { it.key.toString() }))
                ws.send(Wire.pack(ready).toByteString())
                ui("ready", "layers $a-${b - 1}  %.1f ms/layer  loaded in %.1fs".format(msPerLayer, (t0.let { System.nanoTime() - it }) / 1e9))
                startHeartbeat(ws)
            }
            "fwd" -> {
                val c = cfg!!
                val n = hdr.getInt("n"); val pos = hdr.getInt("pos"); val req = hdr.getString("req")
                val x = Wire.decode(payload, hdr.optString("dtype", "bf16"), n * c.hidden)
                val t = System.nanoTime()
                val y = engine.forward(x, n, pos, req)
                val ms = (System.nanoTime() - t) / 1e6
                val outDt = hdr.optString("out_dtype", "bf16")
                val out = JSONObject().put("t", "fwd_out").put("req", req).put("hop", hdr.getInt("hop"))
                    .put("n", n).put("ms", ms).put("dtype", outDt)
                ws.send(Wire.pack(out, Wire.encode(y, outDt)).toByteString())
                forwards++
                onForward()
                val l = layers!!
                ui("layers ${l[0]}-${l[1] - 1}", "req $req  pos $pos  n=$n  %.0f ms   forwards: $forwards".format(ms))
            }
            "fwd_batch" -> {
                val c = cfg!!
                val ra = hdr.getJSONArray("reqs"); val pa = hdr.getJSONArray("pos")
                val reqs = Array(ra.length()) { ra.getString(it) }
                val poss = IntArray(pa.length()) { pa.getInt(it) }
                val x = Wire.decode(payload, hdr.optString("dtype", "bf16"), reqs.size * c.hidden)
                val t = System.nanoTime()
                val y = engine.forwardBatch(x, poss, reqs, c.hidden)
                val ms = (System.nanoTime() - t) / 1e6
                val outDt = hdr.optString("out_dtype", "bf16")
                ws.send(Wire.pack(JSONObject().put("t", "fwd_batch_out").put("key", hdr.getString("key"))
                    .put("batch", reqs.size).put("ms", ms).put("dtype", outDt),
                    Wire.encode(y, outDt)).toByteString())
                forwards++
                onForward()
                val l = layers!!
                ui("layers ${l[0]}-${l[1] - 1}", "batch of ${reqs.size}  %.0f ms   forwards: $forwards".format(ms))
            }
            "reset" -> engine.reset(hdr.getString("req"))
        }
    }

    private fun startHeartbeat(ws: WebSocket) {
        hbThread?.interrupt()
        hbThread = Thread {
            try {
                while (running && !Thread.currentThread().isInterrupted) {
                    val hb = JSONObject().put("t", "hb").put("battery", stats.battery() ?: JSONObject.NULL)
                        .put("thermal", stats.thermal() ?: JSONObject.NULL)
                        .put("cache_reqs", 0).put("rss_mb", stats.rssBytes() / 1048576.0)
                        .put("mem", stats.mem())
                    if (!ws.send(Wire.pack(hb).toByteString())) break
                    Thread.sleep(1000)
                }
            } catch (_: InterruptedException) {}
        }.also { it.isDaemon = true; it.start() }
    }

    private fun fetch(name: String, label: String = name) {
        val dst = File(shardDir, name)
        if (dst.exists() && dst.length() > 0) return
        ui("downloading", label)
        http.newCall(Request.Builder().url("$httpBase/shards/$name").build()).execute().use { r ->
            require(r.isSuccessful) { "GET /shards/$name -> ${r.code}" }
            val tmp = File(shardDir, "$name.part")
            r.body!!.byteStream().use { inp -> tmp.outputStream().use { inp.copyTo(it, 1 shl 20) } }
            tmp.renameTo(dst)
        }
    }

    /** A node keeps only the layers it owns, exactly as the Python nodes do. Both shard kinds are
     *  swept (npz and tflite), so switching a phone between the CPU and NPU runtime never leaves the
     *  other runtime's weights behind. */
    private fun dropForeign(a: Int, b: Int, ext: String) {
        shardDir.listFiles()?.forEach { f ->
            val m = Regex("layer_(\\d\\d)\\.(npz|tflite)").matchEntire(f.name) ?: return@forEach
            val i = m.groupValues[1].toInt()
            if (i < a || i >= b || m.groupValues[2] != ext) f.delete()
        }
    }
}
