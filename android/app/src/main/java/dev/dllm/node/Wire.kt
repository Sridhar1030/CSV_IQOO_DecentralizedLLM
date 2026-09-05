package dev.dllm.node

import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** The frame format the hub speaks: 4-byte big-endian header length, JSON header, raw payload.
 *  Mirrors dllm/wire.py and the codec in dllm/np_node.py exactly, so the hub cannot tell which
 *  runtime a node is. */
object Wire {
    fun pack(hdr: JSONObject, payload: ByteArray = ByteArray(0)): ByteArray {
        val h = hdr.toString().toByteArray(Charsets.UTF_8)
        return ByteBuffer.allocate(4 + h.size + payload.size).order(ByteOrder.BIG_ENDIAN)
            .putInt(h.size).put(h).put(payload).array()
    }

    fun unpack(buf: ByteArray): Pair<JSONObject, ByteArray> {
        val n = ByteBuffer.wrap(buf, 0, 4).order(ByteOrder.BIG_ENDIAN).int
        val hdr = JSONObject(String(buf, 4, n, Charsets.UTF_8))
        return hdr to buf.copyOfRange(4 + n, buf.size)
    }

    /** Activations arrive as bf16 (upper half of an fp32) or, on the last hop, fp32. Little-endian. */
    fun decode(payload: ByteArray, dtype: String, count: Int): FloatArray {
        val out = FloatArray(count)
        val bb = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN)
        if (dtype == "fp32") {
            bb.asFloatBuffer().get(out)
        } else {
            for (i in 0 until count) {
                out[i] = Float.fromBits((bb.getShort(i * 2).toInt() and 0xFFFF) shl 16)
            }
        }
        return out
    }

    fun encode(x: FloatArray, dtype: String): ByteArray {
        if (dtype == "fp32") {
            val bb = ByteBuffer.allocate(x.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            bb.asFloatBuffer().put(x)
            return bb.array()
        }
        // round to nearest even, the same arithmetic as np_node.to_wire
        val bb = ByteBuffer.allocate(x.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in x) {
            val b = v.toRawBits()
            val r = ((b ushr 16) and 1) + 0x7FFF
            bb.putShort(((b + r) ushr 16).toShort())
        }
        return bb.array()
    }
}
