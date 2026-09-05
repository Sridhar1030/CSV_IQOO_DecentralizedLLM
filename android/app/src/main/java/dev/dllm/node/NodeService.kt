package dev.dllm.node

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.net.wifi.WifiManager
import android.os.PowerManager
import java.io.File

/** Runs the node as a foreground service so it survives the screen turning off, which a phone
 *  will do within a minute on stage. Status flows to the activity through [NodeState]. */
class NodeService : Service() {
    private var node: Node? = null
    private var thread: Thread? = null
    private var wake: PowerManager.WakeLock? = null
    private var wifi: WifiManager.WifiLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { stopNode(); stopSelf(); return START_NOT_STICKY }
        val hub = intent?.getStringExtra("hub") ?: return START_NOT_STICKY
        val code = intent.getStringExtra("code") ?: ""
        val name = intent.getStringExtra("name") ?: "phone"
        val engineName = intent.getStringExtra("engine") ?: "cpu"

        startForeground(1, notification("joining $hub"), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "dllm:node").also { it.acquire() }
        // With the screen off, WiFi power saving idles the radio and the socket dies. This keeps it up.
        wifi = (applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager)
            .createWifiLock(if (Build.VERSION.SDK_INT >= 29) WifiManager.WIFI_MODE_FULL_LOW_LATENCY else WifiManager.WIFI_MODE_FULL_HIGH_PERF, "dllm:node")
            .also { it.acquire() }

        stopNode()
        val engine = Engines.byName(engineName)
        NodeState.engine = engine.name
        val n = Node(hub, code, name, engine, File(filesDir, "shards"), Stats(this),
            ui = { s, d -> NodeState.set(s, d); update(notification("$s  $d")) },
            onForward = { NodeState.flash() })
        node = n
        thread = Thread { n.run() }.also { it.start() }
        return START_STICKY
    }

    private fun stopNode() {
        node?.stop(); node = null
        thread?.interrupt(); thread = null
        wake?.let { if (it.isHeld) it.release() }; wake = null
        wifi?.let { if (it.isHeld) it.release() }; wifi = null
        NodeState.set("idle", "")
    }

    override fun onDestroy() { stopNode(); super.onDestroy() }

    private fun notification(text: String): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "dllm node", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("dllm node").setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_share).setContentIntent(open).setOngoing(true).build()
    }

    private fun update(n: Notification) = getSystemService(NotificationManager::class.java).notify(1, n)

    companion object {
        const val CHANNEL = "dllm"
        const val ACTION_STOP = "dev.dllm.node.STOP"
    }
}

/** Tiny observable the activity reads. Avoids pulling in LiveData for two strings. */
object NodeState {
    @Volatile var status = "idle"
    @Volatile var detail = ""
    @Volatile var engine = ""
    @Volatile var listener: ((String, String) -> Unit)? = null
    @Volatile var flashListener: (() -> Unit)? = null
    fun set(s: String, d: String) { status = s; detail = d; listener?.invoke(s, d) }
    fun flash() { flashListener?.invoke() }
}
