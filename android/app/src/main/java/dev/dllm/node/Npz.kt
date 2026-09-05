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
 *  [raw] is the exact bytes numpy wrote (little-endian float32), which is also what the
 *  manifest hash covers. [data] is the same region viewed as floats. Nothing is copied into the
 *  Java heap, so a 500 MB shard costs no heap at all. */
class Tensor(val name: String, val shape: IntArray, val raw: ByteBuffer) {
    val data: FloatBuffer = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val size: Int get() = shape.fold(1) { a, b -> a * b }
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
        require(descr == "<f4") { "$name: dtype $descr, expected little-endian float32" }
        require(Regex("'fortran_order':\\s*(True|False)").find(hdr)!!.groupValues[1] == "False") { "$name: fortran order" }
        val shape = Regex("'shape':\\s*\\(([^)]*)\\)").find(hdr)!!.groupValues[1]
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }.toIntArray()
        val dataStart = headerStart + headerLen
        val db = b.duplicate(); db.position(dataStart)
        val raw = db.slice().order(ByteOrder.LITTLE_ENDIAN)
        return Tensor(name, shape, raw)
    }

    /** Matches dllm.slicer.content_hash byte for byte: sorted keys, then key, dtype, shape,
     *  raw bytes. The hub verifies this against the manifest without holding any weights. */
    fun fingerprint(tensors: Map<String, Tensor>): String {
        val md = MessageDigest.getInstance("SHA-256")
        for (k in tensors.keys.sorted()) {
            val t = tensors[k]!!
            md.update(k.toByteArray())
            md.update("float32".toByteArray())
            val shape = if (t.shape.size == 1) "(${t.shape[0]},)" else t.shape.joinToString(", ", "(", ")")
            md.update(shape.toByteArray())
            md.update(t.raw.duplicate())
        }
        return md.digest().joinToString("") { "%02x".format(it) }.substring(0, 16)
    }
}
