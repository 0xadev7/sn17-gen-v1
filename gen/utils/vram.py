import contextlib, gc, torch


@contextlib.contextmanager
def vram_guard(sync: bool = True, empty: bool = True, ipc_collect: bool = False):
    try:
        yield
    finally:
        # Make sure all Python refs in this scope are collectible
        gc.collect()
        if torch.cuda.is_available():
            if sync:
                torch.cuda.synchronize()
            if empty:
                torch.cuda.empty_cache()
            if ipc_collect:
                # Helpful when many short-lived CUDA allocations occur (e.g., 3D runs)
                torch.cuda.ipc_collect()
