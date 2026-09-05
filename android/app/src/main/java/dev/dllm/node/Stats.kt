package dev.dllm.node

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import org.json.JSONObject

/** What this device can say about itself. The hub stamps it onto this node's hop spans, under
 *  the same attribute names the observability proxy uses, so a trace shows the phone's own RAM,
 *  CPU, battery and heat rather than the coordinator's. Richer than the Termux node can manage:
 *  Android hands us battery and thermal state directly. */
class Stats(private val ctx: Context) {
    private var lastCpuMs = Process.getElapsedCpuTime()
    private var lastWall = SystemClock.elapsedRealtime()

    fun battery(): Int? {
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val v = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        return if (v in 0..100) v else null
    }

    /** 0 none .. 6 shutdown, per PowerManager. */
    fun thermal(): Int? {
        if (Build.VERSION.SDK_INT < 29) return null
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.currentThermalStatus
    }

    fun rssBytes(): Long = Debug.getPss() * 1024L

    fun mem(): JSONObject {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo(); am.getMemoryInfo(mi)
        val cpuMs = Process.getElapsedCpuTime(); val wall = SystemClock.elapsedRealtime()
        val cpu = if (wall > lastWall) 100.0 * (cpuMs - lastCpuMs) / (wall - lastWall) else 0.0
        lastCpuMs = cpuMs; lastWall = wall
        val used = mi.totalMem - mi.availMem
        return JSONObject()
            .put("rss_bytes", rssBytes())
            .put("sys_total_bytes", mi.totalMem)
            .put("sys_used_bytes", used)
            .put("sys_available_bytes", mi.availMem)
            .put("sys_percent", Math.round(1000.0 * used / mi.totalMem) / 10.0)
            .put("cpu_percent", Math.round(cpu * 10) / 10.0)
    }

    fun ramGb(): Double {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo(); am.getMemoryInfo(mi)
        return Math.round(mi.totalMem / 1073741824.0 * 10) / 10.0
    }
}
