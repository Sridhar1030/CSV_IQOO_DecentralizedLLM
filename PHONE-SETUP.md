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

## Proving it is not faking

Anyone can run this. It is the answer to "how do I know one device is not quietly running
the whole model?"

```bash
.venv/bin/python -m dllm.prove
```

Four tests, weakest first.

1. **Every layer lives on exactly one node.** `GET /inventory` collects each node's claimed range
   and checks they tile the model with no gaps and no overlaps.
2. **Each node's claim matches the bytes it holds.** At slice time the builder records a sha256 of
   every layer's weights in `manifest.json`. The hash covers the weights themselves, not the .npz
   container, so it reproduces across independent slicer runs. Each node hashes what it loaded and
   the coordinator compares. The coordinator holds no layer weights at all, and `/inventory` fails
   loudly if it ever does.
3. **Each node's memory is too small for the whole model.** Resident memory is reported by the node
   process and compared with the full checkpoint size.
4. **Remove a node and generation must break.** This is the only test a lying node cannot pass. If
   the remaining devices could still answer, the split was theatre. The check severs the link,
   confirms the request is refused, restores it, and confirms the same answer comes back.

Tests 1 to 3 prove a node does not hold more than it claims. Only test 4 proves the missing layers
are genuinely missing. Lead with test 4 in front of a judge.

### Doing test 4 by hand

```bash
adb kill-server                                   # phone loses the tunnel
curl localhost:8000/v1/chat/completions ...       # 503, pipeline incomplete
adb start-server && adb reverse tcp:8000 tcp:8000 # phone rejoins by itself
```

Nodes reconnect on their own and reclaim the exact range they already hold, so this is repeatable.
Unplugging the cable works the same way and is the better version on stage.

## Where the weights live

| location | holds |
|---|---|
| `mac_shards/` | layers 0-11 only, read by the Mac node |
| `hub_shards/` | the embedding table and lm_head, the tokenizer, and `manifest.json`. No layers |
| phone `~/shards/` | layers 12-23 only |

A node deletes any shard outside its assigned range at startup, so a device that gets reassigned
cannot quietly accumulate the whole model.

One honest caveat. The original Hugging Face checkpoint is still in `~/.cache/huggingface` on the
Mac that ran the slicer, because that is where the slicing happened. For a demo where the claim
must hold under inspection, slice on a machine that is not part of the cluster, or delete the cache
afterwards and keep `manifest.json`, which is all the verification needs.

The coordinator stops being able to serve shards to a new node once its layer copies are gone.
Re-running the slicer restores that ability.
