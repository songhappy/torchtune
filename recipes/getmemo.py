import subprocess
import json

def print_xpu_memory_used_from_xpu_smi(tag, device_id=0):
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", "-d", str(device_id), "-j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        stats = json.loads(result.stdout)
        tile_level = stats.get("tile_level", [])

        total_mem_mb = 0.0
        for tile in tile_level:
            for metric in tile.get("data_list", []):
                if metric["metrics_type"] == "XPUM_STATS_MEMORY_USED":
                    total_mem_mb += metric["value"]

        total_mem_gb = total_mem_mb / 1024
        print(f"[{tag}] xpu-smi total memory used (device {device_id}): {total_mem_gb:.2f} GB")

    except Exception as e:
        print(f"[{tag}] xpu-smi error (device {device_id}): {e}")


print_xpu_memory_used_from_xpu_smi("getmemo")