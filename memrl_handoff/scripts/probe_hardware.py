"""Inventory the machine. Measurements are recorded; nothing is invented."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    return (completed.stdout or "") + (completed.stderr or "")


def main() -> int:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    smi = _run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv",
    ])
    topo = _run(["nvidia-smi", "topo", "-m"])
    (reports / "nvidia-smi.txt").write_text(smi, encoding="utf-8")
    (reports / "topology.txt").write_text(topo, encoding="utf-8")
    usage = shutil.disk_usage(ROOT)
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "nvidia_smi_query": smi,
        "topology": topo,
    }
    try:
        import torch
        payload["torch"] = torch.__version__
        payload["torch_cuda"] = torch.version.cuda
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["bf16_supported"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if torch.cuda.is_available():
            devices = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                    "uuid": str(torch.cuda.get_device_properties(index).uuid) if hasattr(props, "uuid") else None,
                })
            payload["torch_devices"] = devices
            left = torch.randn(256, 256, device="cuda:0", dtype=torch.bfloat16)
            right = torch.randn(256, 256, device="cuda:0", dtype=torch.bfloat16)
            product = left @ right
            payload["bf16_matmul_finite"] = bool(torch.isfinite(product.float()).all().item())
            if torch.cuda.device_count() >= 2:
                payload["ddp"] = _ddp_allreduce()
    except Exception as exc:  # noqa: BLE001 - inventory must record the failure
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    (reports / "hardware.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {reports / 'hardware.json'}")
    return 0


def _ddp_worker(rank: int, world: int, result_path: str) -> None:
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29561"
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(seconds=60))
    tensor = torch.ones(4, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(tensor)
    if rank == 0:
        Path(result_path).write_text(json.dumps({"sum0": float(tensor[0].item())}) + "\n", encoding="utf-8")
    dist.destroy_process_group()


def _ddp_allreduce() -> dict:
    import torch.multiprocessing as mp

    result_path = ROOT / "reports" / "ddp_allreduce.json"
    mp.spawn(_ddp_worker, args=(2, str(result_path)), nprocs=2, join=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
