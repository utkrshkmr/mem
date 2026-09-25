"""T2.14 helper: NVML sampler UUID and rollout power in a job restricted by CUDA_VISIBLE_DEVICES.
usage: CUDA_VISIBLE_DEVICES=2 python nvml_map.py OUT.json ADAPTER_ROOT
"""
import json
import sys
import time

from kmatters.hf_auth import load_hf_token

load_hf_token()


def main():
    OUT, AROOT = sys.argv[1], sys.argv[2]
    from kmatters import config
    from kmatters.engine import reader_sp, train_writer_sp
    from kmatters.loop import build_context, close_context, make_specs
    cfg = config.load(["configs/base.yaml"], ["env.L=4"])
    ctx = build_context(cfg, "nvml_map", AROOT, split="profile", key0=1414, with_trainer=False, nvml=True)
    try:
        time.sleep(5.0)                                   # idle window: engine loaded, nothing running
        t_idle = (time.perf_counter() - 4.5, time.perf_counter())
        pspec, fspec = make_specs(ctx, 0, 32, 4)
        t0 = time.perf_counter()
        ctx.loop.run_until_complete(ctx.executor.collect(pspec, fspec, None, train_writer_sp(ctx.B),
                                                         reader_sp(12), it=0))
        t1 = time.perf_counter()
        out = {"uuid": ctx.nvml.uuid, "idle": ctx.nvml.summary(*t_idle), "rollout": ctx.nvml.summary(t0, t1),
               "rollout_s": t1 - t0}
    finally:
        close_context(ctx)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
