"""Step 1: verify dependencies and a small tensor operation; no data downloads."""
import importlib
import importlib.metadata
import platform
import sys

DEPENDENCIES = {
    "numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
    "scikit-image": "skimage", "matplotlib": "matplotlib", "pillow": "PIL",
    "torch": "torch", "torchvision": "torchvision", "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel", "gradio": "gradio",
}

def main():
    print(f"Python: {platform.python_version()}")
    print(f"System: {platform.system()} ({platform.machine()})")
    print(f"Virtual environment active: {sys.prefix != sys.base_prefix}")
    failed = []
    for package, module in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
            print(f"OK  {package}: {importlib.metadata.version(package)}")
        except Exception as error:
            failed.append(package)
            print(f"FAIL  {package}: {type(error).__name__}: {error}")
    if failed:
        print("\nFix the packages above, then run this check again.")
        return 1

    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    try:
        result = (torch.tensor([1., 2., 3.], device=device) * 2).cpu()
        assert torch.equal(result, torch.tensor([2., 4., 6.]))
    except Exception as error:
        print(f"Tensor check failed on {device}: {error}")
        return 1
    print(f"\nTraining device: {device}")
    print("Tensor operation: PASSED")
    print("STEP 1 CHECKS PASSED. No dataset has been downloaded or model trained.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
