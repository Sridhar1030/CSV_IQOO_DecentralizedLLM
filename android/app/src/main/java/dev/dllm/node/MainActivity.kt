package dev.dllm.node

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView

/** The whole cluster on one phone screen: which device holds which layers, what each is doing this
 *  instant, tokens per second, and the answer as it streams. Every card flashes when a hop lands on
 *  that device, so a row of phones lights up in order as a token moves through them. */
class MainActivity : Activity() {
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var hub: EditText
    private lateinit var code: EditText
    private lateinit var npu: CheckBox
    private lateinit var pipeline: TextView
    private lateinit var tps: TextView
    private lateinit var tokens: TextView
    private lateinit var strip: LinearLayout
    private lateinit var stripLabels: LinearLayout
    private lateinit var nodesBox: LinearLayout
    private lateinit var output: TextView
    private lateinit var selfStatus: TextView
    private lateinit var selfDetail: TextView
    private var watcher: ClusterWatcher? = null
    private var watchedHub = ""
    private val cards = LinkedHashMap<String, Card>()
    private var cells: List<View> = emptyList()

    private val palette = intArrayOf(0xFF1F6FEB.toInt(), 0xFF7EE787.toInt(), 0xFFF2CC60.toInt(), 0xFFFF7B72.toInt(), 0xFFD2A8FF.toInt(), 0xFF79C0FF.toInt())
    private val bg = 0xFF161B22.toInt()
    private val bgFlash = 0xFF1F3B5C.toInt()

    private class Card(val root: LinearLayout, val bar: View, val title: TextView, val doing: TextView, val metrics: TextView, val res: TextView)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)   // a demo phone must not sleep
        hub = findViewById(R.id.hub); code = findViewById(R.id.code); npu = findViewById(R.id.npu)
        pipeline = findViewById(R.id.pipeline); tps = findViewById(R.id.tps); tokens = findViewById(R.id.tokens)
        strip = findViewById(R.id.strip); stripLabels = findViewById(R.id.stripLabels); nodesBox = findViewById(R.id.nodes)
        output = findViewById(R.id.output); selfStatus = findViewById(R.id.selfStatus); selfDetail = findViewById(R.id.selfDetail)

        val prefs = getSharedPreferences("dllm", MODE_PRIVATE)
        hub.setText(prefs.getString("hub", "")); code.setText(prefs.getString("code", ""))
        applyDeepLink(intent)
        if (Build.VERSION.SDK_INT >= 33) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)

        findViewById<Button>(R.id.join).setOnClickListener { join() }
        findViewById<Button>(R.id.leave).setOnClickListener {
            startService(Intent(this, NodeService::class.java).setAction(NodeService.ACTION_STOP))
        }
        selfDetail.text = "runtime cpu-fp32   name ${nodeName()}"
    }

    override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); applyDeepLink(intent) }

    private fun join() {
        val h = hub.text.toString().trim().removePrefix("http://").removeSuffix("/")
        val c = code.text.toString().trim().uppercase()
        if (h.isEmpty()) { selfStatus.text = "hub address needed"; return }
        getSharedPreferences("dllm", MODE_PRIVATE).edit().putString("hub", h).putString("code", c).apply()
        val svc = Intent(this, NodeService::class.java)
            .putExtra("hub", h).putExtra("code", c).putExtra("name", nodeName())
            .putExtra("engine", if (npu.isChecked) "npu" else "cpu")
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(svc) else startService(svc)
        watch(h)
    }

    /** dllm://join?hub=...&code=... from the hub's join page. The person already tapped "open in
     *  the app" there, so this joins straight away rather than asking again. */
    private fun applyDeepLink(intent: Intent?) {
        val d = intent?.data ?: return
        if (d.scheme != "dllm") return
        val h = d.getQueryParameter("hub"); val c = d.getQueryParameter("code")
        h?.let { hub.setText(it) }; c?.let { code.setText(it) }
        // Optional ?npu=1 selects the Hexagon NPU runtime and ?npu=0 forces the CPU one, so a join
        // link picks the engine outright. It has to be able to say "no": the service is
        // START_STICKY, so a phone that once joined on the NPU is restarted by Android with that
        // same intent, and without this there is no way to move it back off tflite shards.
        d.getQueryParameter("npu")?.let { npu.isChecked = it == "1" || it == "true" }
        if (!h.isNullOrBlank() && !c.isNullOrBlank()) join()
    }

    /** Stable per phone, distinct between two phones of the same model. */
    private fun nodeName(): String {
        val id = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID) ?: "0000"
        return "${Build.MODEL.replace(" ", "")}-${id.takeLast(4)}"
    }

    // ------------------------------------------------------------------------------------------
    private fun watch(h: String) {
        if (h == watchedHub && watcher != null) return
        watcher?.stop()
        watchedHub = h
        watcher = ClusterWatcher(h, onState = { s -> ui.post { render(s) } }, onHop = { n -> ui.post { flash(n) } }).also { it.start() }
    }

    override fun onResume() {
        super.onResume()
        val st = Stats(this)
        NodeState.listener = { s, d -> ui.post {
            selfStatus.text = s
            val nsp = st.nspTempC()
            val npuLine = if (nsp != null) "NPU ${nsp}°C  ${if (s == "forward" || s == "ready") "active" else "idle"}" else ""
            selfDetail.text = "$d\nruntime ${NodeState.engine.ifEmpty { "cpu-fp32" }}   name ${nodeName()}${if (npuLine.isNotEmpty()) "\n$npuLine" else ""}"
        } }
        selfStatus.text = NodeState.status; selfDetail.text = NodeState.detail
        val h = hub.text.toString().trim()
        if (h.isNotEmpty()) watch(h)
    }

    override fun onPause() { NodeState.listener = null; watcher?.stop(); watcher = null; watchedHub = ""; super.onPause() }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun render(s: ClusterState) {
        val me = nodeName()
        pipeline.text = when {
            !s.connected -> "hub ${s.hub}  unreachable"
            s.complete -> "hub ${s.hub}   code ${s.code}   pipeline complete"
            else -> "hub ${s.hub}   code ${s.code}   missing layers ${s.missing}"
        }
        pipeline.setTextColor(if (s.complete) 0xFF7EE787.toInt() else 0xFFF2CC60.toInt())
        tps.text = "%.1f".format(s.tokensPerSec)
        tokens.text = "${s.totalTokens} tokens since open"
        output.text = s.output.ifEmpty { "…" }

        // layer strip
        if (cells.size != s.nLayers) {
            strip.removeAllViews(); stripLabels.removeAllViews()
            cells = (0 until s.nLayers).map {
                View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f).also { p -> p.marginEnd = dp(2) }
                    strip.addView(this)
                }
            }
            for (i in 0 until s.nLayers) stripLabels.addView(TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                text = if (i % 4 == 0 || i == s.nLayers - 1) "$i" else ""; textSize = 9f; setTextColor(0xFF5B6773.toInt()); gravity = Gravity.CENTER
            })
        }
        val owner = IntArray(s.nLayers) { -1 }
        s.nodes.forEachIndexed { idx, n -> if (n.live) for (l in n.a until n.b) if (l in owner.indices) owner[l] = idx }
        for (l in 0 until s.nLayers) {
            val idx = owner[l]
            val colour = if (idx < 0) 0xFF21262D.toInt() else palette[idx % palette.size]
            val mine = idx >= 0 && s.nodes[idx].name == me
            cells[l].setBackgroundColor(if (idx < 0 || mine) colour else (colour and 0x00FFFFFF) or 0x99000000.toInt())
        }

        // device cards, updated in place so nothing flickers
        val seen = HashSet<String>()
        s.nodes.forEachIndexed { idx, n ->
            seen.add(n.name)
            val card = cards.getOrPut(n.name) { newCard() }
            val colour = palette[idx % palette.size]
            card.bar.setBackgroundColor(if (n.live) colour else 0xFF30363D.toInt())
            val kind = when {
                n.device.startsWith("android") -> "phone · app"
                n.device.startsWith("phone") -> "phone · termux"
                n.device == "mps" || n.device == "cpu" -> "laptop"
                else -> n.device
            }
            card.title.text = "${n.name}${if (n.name == me) "  (this phone)" else ""}   layers ${n.a}-${n.b - 1}"
            card.title.setTextColor(if (n.live) 0xFFE6EDF3.toInt() else 0xFF5B6773.toInt())
            val age = System.currentTimeMillis() - n.lastSeenMs
            card.doing.text = when {
                !n.live -> "$kind · gone"
                !n.ready -> "$kind · loading"
                n.lastSeenMs > 0 && age < 1500 -> "$kind · ${n.lastPhase}${if (n.lastPhase == "prefill") " ${n.lastN} tokens" else ""}"
                else -> "$kind · idle"
            }
            card.doing.setTextColor(if (n.live && age < 1500 && n.lastSeenMs > 0) 0xFF7EE787.toInt() else 0xFF9AA7B4.toInt())
            card.metrics.text = if (n.hops > 0)
                "last %.0f ms   avg %.0f ms   wire %.0f ms   %d hops".format(n.lastComputeMs, n.avgComputeMs, n.lastWireMs, n.hops)
            else "%.1f ms/layer at join".format(n.msPerLayer)
            val bits = ArrayList<String>()
            n.battery?.let { bits.add("battery $it%") }
            n.sysPercent?.let { if (!it.isNaN()) bits.add("ram %.0f%%".format(it)) }
            n.cpuPercent?.let { if (!it.isNaN()) bits.add("cpu %.0f%%".format(it)) }
            n.rssMb?.let { if (!it.isNaN() && it > 0) bits.add("%.0f MB".format(it)) }
            n.thermal?.let { if (it > 0) bits.add("thermal $it") }
            n.nspTempC?.let { bits.add("NPU %.1f°C".format(it)) }
            card.res.text = bits.joinToString("   ")
        }
        for (k in cards.keys.toList()) if (k !in seen) { nodesBox.removeView(cards[k]!!.root); cards.remove(k) }
        // keep card order by layer
        s.nodes.forEachIndexed { i, n -> cards[n.name]?.let { c -> if (nodesBox.indexOfChild(c.root) != i) { nodesBox.removeView(c.root); nodesBox.addView(c.root, minOf(i, nodesBox.childCount)) } } }
    }

    private fun newCard(): Card {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; setBackgroundColor(bg)
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).also { it.bottomMargin = dp(8) }
        }
        val bar = View(this).apply { layoutParams = LinearLayout.LayoutParams(dp(6), LinearLayout.LayoutParams.MATCH_PARENT) }
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        fun tv(size: Float, colour: Int, mono: Boolean = true) = TextView(this).apply {
            textSize = size; setTextColor(colour); if (mono) typeface = Typeface.MONOSPACE
        }
        val title = tv(15f, 0xFFE6EDF3.toInt()).apply { setTypeface(Typeface.MONOSPACE, Typeface.BOLD) }
        val doing = tv(13f, 0xFF9AA7B4.toInt())
        val metrics = tv(12f, 0xFF9AA7B4.toInt())
        val res = tv(12f, 0xFF5B6773.toInt())
        col.addView(title); col.addView(doing); col.addView(metrics); col.addView(res)
        root.addView(bar); root.addView(col)
        nodesBox.addView(root)
        return Card(root, bar, title, doing, metrics, res)
    }

    private fun flash(name: String) {
        val c = cards[name] ?: return
        c.root.setBackgroundColor(bgFlash)
        ui.postDelayed({ c.root.setBackgroundColor(bg) }, 110)
    }
}
