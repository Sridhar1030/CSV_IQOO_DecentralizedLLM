#!/bin/bash
# Stage the NPU runtime the app packages: android/app/src/main/jniLibs/arm64-v8a/.
#
# Two sources, neither needing a Qualcomm account:
#   * LiteRT NIGHTLY from Google's public GCS bucket. Nightly, not "latest" and not the Maven
#     release: those SIGBUS on the Snapdragon 8 Elite Gen 5 (see how_torun_on_npu.md).
#   * QAIRT 2.49.0 QNN libraries. Five come from the Maven AAR; libQnnIr.so and libQnnSaver.so are
#     missing there, so they are pulled out of the 2.35 GB QAIRT zip with HTTP range reads of its
#     central directory, which takes seconds and downloads only those two members.
#
# Then run npu/build_jni.sh to compile the JNI bridge against what this staged.
set -e
J=android/app/src/main/jniLibs/arm64-v8a
mkdir -p "$J"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "[1/3] LiteRT nightly (arm64)"
for f in libLiteRt.so libLiteRtDispatch_Qualcomm.so libLiteRtCompilerPlugin_Qualcomm.so; do
  curl -sSL -o "$J/$f" "https://storage.googleapis.com/litert/binaries/nightly/android_arm64/$f"
  echo "      $f  $(stat -f%z "$J/$f" 2>/dev/null || stat -c%s "$J/$f") bytes"
done

echo "[2/3] QNN 2.49.0 from Maven (libQnnHtp, HtpPrepare, HtpV81Skel, HtpV81Stub, System)"
curl -sSL -o "$TMP/qnn.aar" \
  "https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/2.49.0/qnn-runtime-2.49.0.aar"
for f in libQnnHtp.so libQnnHtpPrepare.so libQnnHtpV81Skel.so libQnnHtpV81Stub.so libQnnSystem.so; do
  unzip -p "$TMP/qnn.aar" "jni/arm64-v8a/$f" > "$J/$f"
  echo "      $f  $(stat -f%z "$J/$f" 2>/dev/null || stat -c%s "$J/$f") bytes"
done

echo "[3/3] libQnnIr.so + libQnnSaver.so by range read of the QAIRT zip"
python3 - "$J" <<'PY'
import os, struct, sys, urllib.request, zlib
OUT = sys.argv[1]
URL = ('https://softwarecenter.qualcomm.com/api/download/software/sdks/'
       'Qualcomm_AI_Runtime_Community/All/2.49.0.260730/v2.49.0.260730.zip')
WANT = {'qairt/2.49.0.260730/lib/aarch64-android/libQnnIr.so',
        'qairt/2.49.0.260730/lib/aarch64-android/libQnnSaver.so'}

def rng(a, b):
    r = urllib.request.Request(URL, headers={'Range': f'bytes={a}-{b}'})
    return urllib.request.urlopen(r, timeout=180).read()

size = int(urllib.request.urlopen(urllib.request.Request(URL, headers={'Range': 'bytes=0-0'}),
                                  timeout=180).headers['Content-Range'].split('/')[1])
tail = rng(size - 70000, size - 1)
j = tail.rfind(b'PK\x05\x06')
cd_size, cd_off = struct.unpack_from('<II', tail, j + 12)
cd, p = rng(cd_off, cd_off + cd_size - 1), 0
while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
    method = struct.unpack_from('<H', cd, p + 10)[0]
    csz, = struct.unpack_from('<I', cd, p + 20)
    nlen, elen, clen = struct.unpack_from('<HHH', cd, p + 28)
    lho, = struct.unpack_from('<I', cd, p + 42)
    nm = cd[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
    if nm in WANT:
        lh = rng(lho, lho + 29)
        ln, le = struct.unpack_from('<HH', lh, 26)
        data = rng(lho + 30 + ln + le, lho + 30 + ln + le + csz - 1)
        raw = zlib.decompress(data, -15) if method == 8 else data
        dst = os.path.join(OUT, os.path.basename(nm))
        open(dst, 'wb').write(raw)
        print(f"      {os.path.basename(nm)}  {len(raw)} bytes")
    p += 46 + nlen + elen + clen
PY

echo
echo "staged $(ls "$J" | wc -l | tr -d ' ') libraries in $J"
echo "next: npu/build_jni.sh"
