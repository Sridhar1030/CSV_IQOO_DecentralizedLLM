# Phone node setup (Android, Termux, USB)

The phone runs one Python file with numpy only. No app build, no torch, no safetensors.
It holds a contiguous range of layers and nothing else. The weights for those layers are
downloaded from the Mac at join time.

## One time, on the phone

1. Install **Termux from F-Droid**: https://f-droid.org/packages/com.termux/
   The Play Store build is abandoned and its packages fail to install. Use F-Droid.
2. Open Termux:
   ```
   pkg update -y && pkg install -y python
   pip install numpy websockets
   ```
   `pip install numpy` builds from source and takes a few minutes. Let it finish.
3. Settings on the phone: Developer options on, **USB debugging** on. Plug in USB-C.
   On first connect tap **Allow USB debugging** and tick "always allow from this computer".

## One time, on the Mac

```bash
brew install android-platform-tools
adb devices          # must print "device", not "unauthorized"
```

## Every run

Mac, terminal 1 — hub. Prints the lobby code.

```bash
.venv/bin/python -m dllm.hub --shards shards --expected 2 --port 8000
```

Mac, terminal 2 — forward the phone's localhost:8000 to the Mac's hub, then the Mac's own node.

```bash
adb reverse tcp:8000 tcp:8000
.venv/bin/python -m dllm.node --code <LOBBY> --name mac1 --layers 0-12 --device mps
```

Phone, in Termux — fetch the node source from the hub and join.

```bash
curl -sO http://127.0.0.1:8000/node.py
python node.py --hub ws://127.0.0.1:8000/ws/node --code <LOBBY> --name phoneA --layers 12-24
```

The phone downloads only `layer_12.npz` through `layer_23.npz`. It never sees the other layers,
the embedding table or the lm_head.

Mac, terminal 3 — talk to it.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":32}'
```

`adb reverse` makes the phone's own `127.0.0.1:8000` point at the Mac over the USB cable. No WiFi,
no router, no IP addresses to type. For the 4-device demo, drop `adb reverse` and give every node
the Mac's LAN IP instead.

## Checks

- `curl localhost:8000/status` — which node holds which layers, ms per layer, battery.
- `curl -N localhost:8000/events` — live stream of every hop, with compute and wire time split out.
