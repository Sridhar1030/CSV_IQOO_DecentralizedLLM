package dev.dllm.node

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import java.util.zip.Inflater

/** One tensor from a .npz, memory-mapped straight out of the file.
 *  [raw] is the exact bytes numpy wrote, which is also what the manifest hash covers. [data] is
 *  the same region viewed as floats, valid only for an fp32 tensor. Nothing is copied into the
 *  Java heap, so a 500 MB shard costs no heap at all.
 *
 *  [dtype] is numpy's own descr: "f4" fp32, "i1" int8, "u1" int4 packed two codes per byte.
 *  A quantised tensor carries its [scale] alongside, one fp32 per output row for int8 and one
 *  per group of columns for int4. [cols] is the logical width, which for int4 is twice the
 *  width of the bytes on disk. */
class Tensor(val name: String, val shape: IntArray, val raw: ByteBuffer, val dtype: String = "f4") {
    var scale: Tensor? = null
    val data: FloatBuffer get() = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val rows: Int get() = shape[0]
    val cols: Int get() = if (dtype == "u1") shape[1] * 2 else shape[1]
    val size: Int get() = shape.fold(1) { a, b -> a * b }
}

/** Pulls one output row out of a tensor as fp32, whatever the shard stores it as, so the hot loop
 *  below stays a plain float dot product. One per thread: it holds its own buffer positions. */
class RowReader(private val t: Tensor) {
    private val buf: ByteBuffer = t.raw.duplicate().order(ByteOrder.LITTLE_ENDIAN)
    private val floats: FloatBuffer = buf.asFloatBuffer()
    private val sc: FloatBuffer? = t.scale?.raw?.duplicate()?.order(ByteOrder.LITTLE_ENDIAN)?.asFloatBuffer()
    private val packed = ByteArray(if (t.dtype == "u1") t.cols / 2 else 0)
    private val groups = if (t.dtype == "u1") t.scale!!.shape[1] else 0

    fun read(o: Int, dst: FloatArray) {
        val cols = t.cols
        when (t.dtype) {
            "f4" -> { floats.position(o * cols); floats.get(dst, 0, cols) }
            "i1" -> {
                buf.position(o * cols)
                val s = sc!!.get(o)
                for (i in 0 until cols) dst[i] = buf.get().toInt() * s
            }
            else -> {
                val half = cols / 2
                buf.position(o * half); buf.get(packed, 0, half)
                val g = cols / groups
                var i = 0
                for (gi in 0 until groups) {
                    val s = sc!!.get(o * groups + gi)
                    val end = i + g
                    while (i < end) {
                        val v = packed[i / 2].toInt() and 0xFF
                        dst[i] = ((v and 0x0F) - 8) * s
                        dst[i + 1] = ((v shr 4) - 8) * s
                        i += 2
                    }
                }
            }
        }
    }
}

/** Reads the .npz files dllm/slicer.py writes. np.savez stores entries uncompressed, so each
 *  array's bytes sit contiguously inside the zip and can be mapped in place. Deflated entries
 *  are handled too, into native memory. No numpy, no extraction step. */
object Npz {
    fun load(file: File): Map<String, Tensor> {
        RandomAccessFile(file, "r").use { raf ->
            val ch = raf.channel
            val size = raf.length()
            // End-of-central-directory record: scan back for its signature.
            val tailLen = minOf(size, 66_000L).toInt()
            val tail = ByteBuffer.allocate(tailLen).order(ByteOrder.LITTLE_ENDIAN)
            ch.read(tail, size - tailLen); tail.flip()
            var eocd = -1
            for (i in tailLen - 22 downTo 0) {
                if (tail.getInt(i) == 0x06054b50) { eocd = i; break }
            }
            require(eocd >= 0) { "${file.name}: not a zip file" }
            val entries = tail.getShort(eocd + 10).toInt() and 0xFFFF
            val cdOffset = tail.getInt(eocd + 16).toLong() and 0xFFFFFFFFL
            require(cdOffset != 0xFFFFFFFFL) { "${file.name}: zip64 archives are not expected here" }

            val out = HashMap<String, Tensor>()
            var p = cdOffset
            repeat(entries) {
                val cd = readAt(ch, p, 46)
                require(cd.getInt(0) == 0x02014b50) { "${file.name}: bad central directory" }
                val method = cd.getShort(10).toInt() and 0xFFFF
                val compSize = cd.getInt(20).toLong() and 0xFFFFFFFFL
                val uncompSize = cd.getInt(24).toLong() and 0xFFFFFFFFL
                val nameLen = cd.getShort(28).toInt() and 0xFFFF
                val extraLen = cd.getShort(30).toInt() and 0xFFFF
                val commentLen = cd.getShort(32).toInt() and 0xFFFF
                val localOff = cd.getInt(42).toLong() and 0xFFFFFFFFL
                val name = String(readAt(ch, p + 46, nameLen).array(), Charsets.UTF_8)
                p += 46 + nameLen + extraLen + commentLen

                val lh = readAt(ch, localOff, 30)
                require(lh.getInt(0) == 0x04034b50) { "${file.name}: bad local header for $name" }
                val dataOff = localOff + 30 + (lh.getShort(26).toInt() and 0xFFFF) + (lh.getShort(28).toInt() and 0xFFFF)

                val bytes: ByteBuffer = if (method == 0) {
                    ch.map(FileChannel.MapMode.READ_ONLY, dataOff, uncompSize)
                } else {
                    val comp = readAt(ch, dataOff, compSize.toInt()).array()
                    val inf = Inflater(true); inf.setInput(comp)
                    val dst = ByteArray(uncompSize.toInt()); var got = 0
                    while (got < dst.size && !inf.finished()) got += inf.inflate(dst, got, dst.size - got)
                    inf.end()
                    ByteBuffer.allocateDirect(dst.size).put(dst).also { it.flip() }
                }
                val key = name.removeSuffix(".npy")
                out[key] = npy(key, bytes.order(ByteOrder.LITTLE_ENDIAN))
            }
            // A quantised tensor keeps its own key and stores its scales under "<key>.scale".
            // Hang each scale off its tensor so a reader needs only the one lookup; the scale
            // entries stay in the map too, because the manifest hash covers every array.
            for ((k, t) in out) if (k.endsWith(".scale")) out[k.removeSuffix(".scale")]?.scale = t
            return out
        }
    }

    private fun readAt(ch: FileChannel, pos: Long, len: Int): ByteBuffer {
        val b = ByteBuffer.allocate(len).order(ByteOrder.LITTLE_ENDIAN)
        var off = 0
        while (off < len) { val n = ch.read(b, pos + off); require(n > 0); off += n }
        b.flip(); return b
    }

    /** .npy: magic, version, header dict as text, then the raw array. */
    private fun npy(name: String, b: ByteBuffer): Tensor {
        require(b.get(0) == 0x93.toByte() && b.get(1) == 'N'.code.toByte()) { "$name: not .npy" }
        val major = b.get(6).toInt()
        val headerLen: Int; val headerStart: Int
        if (major == 1) { headerLen = b.getShort(8).toInt() and 0xFFFF; headerStart = 10 }
        else { headerLen = b.getInt(8); headerStart = 12 }
        // Buffer.position(int) returns the base Buffer type on Android, so no chaining off it.
        val hdrBytes = ByteArray(headerLen)
        val hb = b.duplicate(); hb.position(headerStart); hb.get(hdrBytes)
        val hdr = String(hdrBytes, Charsets.ISO_8859_1)
        val descr = Regex("'descr':\\s*'([^']+)'").find(hdr)!!.groupValues[1]
        val dtype = when (descr) {
            "<f4" -> "f4"
            "|i1" -> "i1"                                 // int8, one scale per output row
            "|u1" -> "u1"                                 // int4, two codes per byte
            else -> throw IllegalArgumentException("$name: dtype $descr is not one this node reads")
        }
        require(Regex("'fortran_order':\\s*(True|False)").find(hdr)!!.groupValues[1] == "False") { "$name: fortran order" }
        val shape = Regex("'shape':\\s*\\(([^)]*)\\)").find(hdr)!!.groupValues[1]
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }.toIntArray()
        val dataStart = headerStart + headerLen
        val db = b.duplicate(); db.position(dataStart)
        val raw = db.slice().order(ByteOrder.LITTLE_ENDIAN)
        return Tensor(name, shape, raw, dtype)
    }

    /** Matches dllm.slicer.content_hash byte for byte: sorted keys, then key, dtype, shape,
     *  raw bytes. The hub verifies this against the manifest without holding any weights. */
    fun fingerprint(tensors: Map<String, Tensor>): String {
        val md = MessageDigest.getInstance("SHA-256")
        for (k in tensors.keys.sorted()) {
            val t = tensors[k]!!
            md.update(k.toByteArray())
            md.update(when (t.dtype) { "i1" -> "int8"; "u1" -> "uint8"; else -> "float32" }.toByteArray())
            val shape = if (t.shape.size == 1) "(${t.shape[0]},)" else t.shape.joinToString(", ", "(", ")")
            md.update(shape.toByteArray())
            md.update(t.raw.duplicate())
        }
        return md.digest().joinToString("") { "%02x".format(it) }.substring(0, 16)
    }
}
