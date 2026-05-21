import torch
import time
import multiprocessing as mp
import random

def gpu_worker(gpu_id,
               base_ratio=0.72,
               fluctuation=0.10,   # 波动更大，但有硬限制
               hard_cap=0.90,      # 👈 显存绝对上限
               safety_buffer_gb=3, # 👈 额外预留（关键）
               matrix_size=4096,
               dtype=torch.float16):

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory

    safety_buffer = int(safety_buffer_gb * 1024**3)
    max_allowed = int(total_mem * hard_cap) - safety_buffer

    print(f"[GPU {gpu_id}] Total: {total_mem/1024**3:.1f} GB")
    print(f"[GPU {gpu_id}] Hard cap: {hard_cap*100:.0f}% - {safety_buffer_gb}GB buffer")

    buffers = []
    allocated = 0

    chunk_size = 256 * 1024 * 1024  # 256MB

    def allocate_to_target(target_bytes):
        nonlocal allocated
        while allocated < target_bytes:
            try:
                t = torch.empty(chunk_size // 2, dtype=dtype, device=device)
                buffers.append(t)
                allocated += t.element_size() * t.nelement()
            except RuntimeError:
                break

    def free_some_memory(target_bytes):
        nonlocal allocated
        while allocated > target_bytes and buffers:
            t = buffers.pop()
            allocated -= t.element_size() * t.nelement()
            del t
        torch.cuda.empty_cache()

    # ===== 初始化 =====
    init_target = int(total_mem * base_ratio)
    allocate_to_target(init_target)

    print(f"[GPU {gpu_id}] Init: {allocated/1024**3:.2f} GB")

    # ===== 计算矩阵 =====
    A = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    B = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)

    iteration = 0

    while True:
        # ===== 目标显存（带硬限制）=====
        ratio = base_ratio + random.uniform(-fluctuation, fluctuation)
        target_mem = int(total_mem * ratio)

        # 👇 强制限制：不超过安全上限
        target_mem = min(target_mem, max_allowed)

        # 再加一个下界（防太低）
        min_allowed = int(total_mem * 0.65)
        target_mem = max(target_mem, min_allowed)

        # ===== 动态调整 =====
        if allocated < target_mem:
            allocate_to_target(target_mem)
        else:
            free_some_memory(target_mem)

        # ===== 利用率波动 =====
        compute_repeat = random.randint(2, 6)

        start = time.time()

        for _ in range(compute_repeat):
            C = torch.matmul(A, B)

        torch.cuda.synchronize()

        # ===== 实时监控（防突发OOM）=====
        current_alloc = torch.cuda.memory_allocated(device)

        # 如果接近上限，强制释放一点
        if current_alloc > max_allowed:
            free_some_memory(int(max_allowed * 0.95))
            print(f"[GPU {gpu_id}] ⚠️ Hit cap, force release")

        iteration += 1

        print(f"[GPU {gpu_id}] Iter {iteration} | "
              f"Mem: {allocated/1024**3:.2f} GB | "
              f"Util x{compute_repeat} | "
              f"{time.time()-start:.3f}s")

        time.sleep(random.uniform(0.0, 0.2))


def parse_gpu_ids(value: str):
    """Parse a comma-separated list of GPU ids, e.g. '0,1,2,3'."""
    if value is None:
        return []
    items = [x.strip() for x in value.split(",") if x.strip() != ""]
    return [int(x) for x in items]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Occupy GPUs with synthetic workload.")
    parser.add_argument(
        "--gpus",
        "--gpu_ids",
        dest="gpus",
        type=parse_gpu_ids,
        default=[6, 7],
        help="Comma-separated GPU ids, e.g. --gpus=0,1,2,3,4",
    )
    args = parser.parse_args()

    gpu_ids = args.gpus
    if not gpu_ids:
        raise SystemExit("No GPU ids provided. Use --gpus=0,1,...")

    print(f"[main] Launching workers on GPUs: {gpu_ids}")

    # Use 'spawn' to avoid CUDA fork issues — the top-level `import torch`
    # initializes CUDA, which is not fork-safe.  'spawn' ensures each worker
    # gets a fresh CUDA context.
    mp.set_start_method('spawn', force=True)

    processes = []
    for gid in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gid,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()