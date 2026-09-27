"""
GPU-idle gate.

Training only proceeds when the GPU is idle (low utilization + enough free
VRAM). Designed to be a good GPU citizen on a shared laptop:
  * transient nvidia-smi failures are treated as BUSY (keep waiting), never as
    idle -- a query typically fails *because* the GPU/driver is saturated;
  * the gate is meant to be polled every epoch (cheap single probe when idle,
    sustained wait only when busy);
  * the trainer releases its cached VRAM while paused (see train.py), so a
    paused run does not hog memory the user wants.

Idle is only assumed if nvidia-smi was NEVER reachable (i.e. truly absent).

Tunable via env vars:
  ESC_GPU_UTIL_MAX   max GPU util % to consider idle        (default 25)
  ESC_GPU_MIN_FREE   min free VRAM (MiB) required           (default 3000)
  ESC_GPU_SUSTAIN    seconds util must stay low             (default 15)
  ESC_GPU_POLL       poll interval seconds                  (default 5)
  ESC_GPU_TIMEOUT    give-up seconds (0 = wait forever)     (default 0)
  ESC_GPU_DISABLE    if "1", skip the gate entirely         (default 0)
"""
import os
import time
import subprocess

_ever_ok = False   # have we EVER gotten a successful nvidia-smi reading?


def _query_once():
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=utilization.gpu,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        stderr=subprocess.STDOUT, timeout=15,
    ).decode().strip().splitlines()[0]
    util, used, free = [int(x.strip()) for x in out.split(",")]
    return util, used, free


def _query(retries=3):
    """Return (util, used, free) or None. Retries to smooth transient failures."""
    global _ever_ok
    for i in range(retries):
        try:
            q = _query_once()
            _ever_ok = True
            return q
        except Exception:
            if i < retries - 1:
                time.sleep(1.0)
    return None


def snapshot():
    q = _query(retries=1)
    if q is None:
        return "GPU status unavailable"
    return f"util={q[0]}%  used={q[1]}MiB  free={q[2]}MiB"


def _thresholds(util_max, min_free):
    util_max = int(os.environ.get("ESC_GPU_UTIL_MAX", 25)) if util_max is None else util_max
    min_free = int(os.environ.get("ESC_GPU_MIN_FREE", 3000)) if min_free is None else min_free
    return util_max, min_free


def free_mib():
    """Instantaneous free VRAM in MiB, or None on failure. Not affected by our
    own compute util -- only by memory actually allocated (ours + others)."""
    q = _query()
    return None if q is None else q[2]


def external_busy(our_reserved_mib=0.0, ext_max=None):
    """
    True if ANOTHER GPU app (a game / other training) is using significant VRAM.

    We estimate memory used by everyone-but-us as (total used - OUR reserved),
    so our own multi-GB training footprint and 100% util never count as "busy"
    -- otherwise the gate would fight itself. `our_reserved_mib` should be
    torch.cuda.memory_reserved()/MiB. Pause when the external estimate exceeds
    ESC_GPU_EXT_MAX (default 4000 MiB; covers the ~2-2.5 GB desktop baseline plus
    headroom, so only a genuinely heavy new app trips it).
    Transient nvidia-smi failure -> not busy (proceed; a real OOM is retried).
    """
    if os.environ.get("ESC_GPU_DISABLE", "0") == "1":
        return False
    ext_max = int(os.environ.get("ESC_GPU_EXT_MAX", 4000)) if ext_max is None else ext_max
    q = _query()
    if q is None:
        return False
    util, used, free = q
    external = used - our_reserved_mib
    return external > ext_max


def is_idle_now(util_max=None, min_free=None):
    """Single probe. True only if genuinely idle. Transient failure -> busy."""
    if os.environ.get("ESC_GPU_DISABLE", "0") == "1":
        return True
    util_max, min_free = _thresholds(util_max, min_free)
    q = _query()
    if q is None:
        # nvidia-smi failed. Absent -> assume idle; otherwise treat as busy.
        return not _ever_ok
    util, used, free = q
    return (util <= util_max) and (free >= min_free)


def wait_for_idle(min_free=None, sustain=None, poll=None, timeout=None, label=""):
    """
    Wait until the GPU has enough FREE VRAM (resume threshold, default 3000 MiB)
    for a sustained window -- i.e. any big external app has released the GPU.
    Gating on free memory (not util) avoids counting our own training util as
    "busy". util is still shown in logs for context.
    """
    if os.environ.get("ESC_GPU_DISABLE", "0") == "1":
        print(f"[gpu-gate] disabled by ESC_GPU_DISABLE=1 -> proceeding {label}", flush=True)
        return True

    _, min_free = _thresholds(None, min_free)
    sustain = int(os.environ.get("ESC_GPU_SUSTAIN", 15)) if sustain is None else sustain
    poll = int(os.environ.get("ESC_GPU_POLL", 5)) if poll is None else poll
    timeout = int(os.environ.get("ESC_GPU_TIMEOUT", 0)) if timeout is None else timeout

    print(f"[gpu-gate] waiting {label}: need free>={min_free}MiB sustained {sustain}s", flush=True)
    start = time.monotonic()
    idle_since = None
    while True:
        q = _query()
        now = time.monotonic()
        if q is None:
            if not _ever_ok:
                print(f"[gpu-gate] nvidia-smi never reachable -> assuming idle {label}", flush=True)
                return True
            idle_since = None
            print("[gpu-gate] nvidia-smi query failed - treating as BUSY, waiting...", flush=True)
        else:
            util, used, free = q
            if free >= min_free:
                if idle_since is None:
                    idle_since = now
                held = now - idle_since
                if held >= sustain:
                    print(f"[gpu-gate] free VRAM ok ({snapshot()}) -> proceeding {label}", flush=True)
                    return True
                print(f"[gpu-gate] free {held:.0f}/{sustain}s (free={free}MiB util={util}%)", flush=True)
            else:
                if idle_since is not None:
                    print(f"[gpu-gate] busy again (free={free}MiB) - resetting", flush=True)
                idle_since = None
                print(f"[gpu-gate] external app using GPU (used={used}MiB free={free}MiB util={util}%) - waiting...", flush=True)
        if timeout and (now - start) > timeout:
            print(f"[gpu-gate] timeout after {timeout}s -> proceeding anyway {label}", flush=True)
            return True
        time.sleep(poll)


if __name__ == "__main__":
    print("[gpu-gate] current:", snapshot())
    print("[gpu-gate] is_idle_now:", is_idle_now())
