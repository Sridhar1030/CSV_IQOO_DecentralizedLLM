package dev.dllm.node

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.ArrayDeque
import java.util.concurrent.TimeUnit

/** One node as the dashboard sees it: what the hub says about it, plus what the live event stream
 *  has shown it doing. */
class NodeView(
    val name: String, val device: String, val a: Int, val b: Int,
    val live: Boolean, val ready: Boolean, val msPerLayer: Double,
    val battery: Int?, val sysPercent: Double?, val cpuPercent: Double?, val rssMb: Double?, val thermal: Int?, val nspTempC: Double?,
    val hops: Long, val lastComputeMs: Double, val avgComputeMs: Double, val lastWireMs: Double,
    val lastPhase: String, val lastN: Int, val lastSeenMs: Long,
)

class ClusterState(
    val hub: String, val connected: Boolean, val code: String, val nLayers: Int,
    val complete: Boolean, val missing: String, val nodes: List<NodeView>,
    val tokensPerSec: Double, val totalTokens: Long, val lastTokenMs: Long, val output: String,
)

/** Watches the whole cluster from any phone: polls the hub's /status once a second for who holds
 *  which layers, and holds its /events stream open for every hop and every token as they happen.
 *  Pure presentation, so it lives with the activity, not the service. */
class ClusterWatcher(private val hub: String, private val onState: (ClusterState) -> Unit, private val onHop: (String) -> Unit) {
    private val http = OkHttpClient.Builder().readTimeout(0, TimeUnit.MILLISECONDS).connectTimeout(4, TimeUnit.SECONDS).build()
    @Volatile private var running = true
    private val lock = Any()
    private var status: JSONObject? = null
    private var connected = false
    private val tokenTimes = ArrayDeque<Long>()
    private var totalTokens = 0L
    private var lastTokenMs = 0L
    private val output = StringBuilder()
    private var lastReq = ""
    private var lastPublish = 0L

    private class HopStats { var hops = 0L; var last = 0.0; var avg = 0.0; var wire = 0.0; var phase = "idle"; var n = 0; var seen = 0L }
    private val hopStats = HashMap<String, HopStats>()

    fun start() {
        Thread { pollLoop() }.also { it.isDaemon = true; it.name = "dllm-poll" }.start()
        Thread { eventLoop() }.also { it.isDaemon = true; it.name = "dllm-events" }.start()
    }
    fun stop() { running = false }

    private fun pollLoop() {
        while (running) {
            try {
                http.newCall(Request.Builder().url("http://$hub/status").build()).execute().use { r ->
                    val j = JSONObject(r.body!!.string())
                    synchronized(lock) { status = j; connected = true }
                }
            } catch (e: Exception) {
                synchronized(lock) { connected = false }
            }
            publish(force = true)
            try { Thread.sleep(1000) } catch (_: InterruptedException) { return }
        }
    }

    private fun eventLoop() {
        while (running) {
            try {
                http.newCall(Request.Builder().url("http://$hub/events").build()).execute().use { r ->
                    val src = r.body!!.source()
                    while (running) {
                        val line = src.readUtf8Line() ?: break
                        if (line.startsWith("data: ")) onEvent(JSONObject(line.substring(6)))
                    }
                }
            } catch (e: Exception) {
                Log.i("dllm", "events stream: $e")
            }
            if (running) try { Thread.sleep(2000) } catch (_: InterruptedException) { return }
        }
    }

    private fun onEvent(e: JSONObject) {
        val now = System.currentTimeMillis()
        var hopNode: String? = null
        synchronized(lock) {
            when (e.optString("t")) {
                "token" -> {
                    tokenTimes.addLast(now); totalTokens++; lastTokenMs = now
                    while (tokenTimes.isNotEmpty() && now - tokenTimes.first() > 5000) tokenTimes.removeFirst()
                    val req = e.optString("req")
                    if (req != lastReq) { output.setLength(0); lastReq = req }
                    output.append(e.optString("text"))
                    if (output.length > 400) output.delete(0, output.length - 400)
                }
                "hop" -> {
                    val name = e.getString("node"); hopNode = name
                    val s = hopStats.getOrPut(name) { HopStats() }
                    s.hops++; s.last = e.getDouble("compute_ms"); s.wire = e.getDouble("wire_ms")
                    s.avg = if (s.hops == 1L) s.last else s.avg * 0.9 + s.last * 0.1
                    s.n = e.getInt("n"); s.phase = if (s.n > 1) "prefill" else "decode"; s.seen = now
                }
            }
        }
        hopNode?.let(onHop)
        publish(force = false)
    }

    private fun publish(force: Boolean) {
        val now = System.currentTimeMillis()
        if (!force && now - lastPublish < 80) return
        lastPublish = now
        val st: ClusterState
        synchronized(lock) {
            val j = status
            val nodes = ArrayList<NodeView>()
            if (j != null) {
                val ns = j.getJSONObject("nodes")
                for (k in ns.keys()) {
                    val v = ns.getJSONObject(k)
                    val layers = v.getJSONArray("layers")
                    val mem = v.optJSONObject("mem")
                    val h = hopStats[k]
                    nodes.add(NodeView(
                        name = k, device = v.optString("device", "?"), a = layers.getInt(0), b = layers.getInt(1),
                        live = v.optBoolean("live", false), ready = v.optBoolean("ready", false),
                        msPerLayer = v.optDouble("ms_per_layer", 0.0),
                        battery = if (v.isNull("battery")) null else v.optInt("battery"),
                        sysPercent = mem?.optDouble("sys_percent"), cpuPercent = mem?.optDouble("cpu_percent"),
                        rssMb = mem?.let { it.optDouble("rss_bytes") / 1048576.0 },
                        thermal = if (v.has("thermal") && !v.isNull("thermal")) v.optInt("thermal") else null,
                        nspTempC = if (v.has("nsp_temp_c") && !v.isNull("nsp_temp_c")) v.optDouble("nsp_temp_c") else null,
                        hops = h?.hops ?: 0, lastComputeMs = h?.last ?: 0.0, avgComputeMs = h?.avg ?: 0.0,
                        lastWireMs = h?.wire ?: 0.0, lastPhase = h?.phase ?: "idle", lastN = h?.n ?: 0, lastSeenMs = h?.seen ?: 0,
                    ))
                }
            }
            nodes.sortBy { it.a }
            val tps = tokenTimes.count { now - it <= 5000 } / 5.0
            st = ClusterState(
                hub = hub, connected = connected, code = j?.optString("code") ?: "",
                nLayers = j?.optInt("n_layers") ?: 24, complete = j?.optBoolean("pipeline_ok") ?: false,
                missing = j?.optString("missing_layers") ?: "", nodes = nodes,
                tokensPerSec = tps, totalTokens = totalTokens, lastTokenMs = lastTokenMs, output = output.toString(),
            )
        }
        onState(st)
    }
}
