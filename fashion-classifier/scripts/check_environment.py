"""Check imports and synthetic tensor operations; never downloads data."""
import importlib
from importlib import metadata
import platform
import sys

PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "torch": "torch", "torchvision": "torchvision",
    "scikit-learn": "sklearn", "scikit-image": "skimage",
    "matplotlib": "matplotlib", "gradio": "gradio", "Pillow": "PIL",
    "jupyterlab": "jupyterlab", "ipykernel": "ipykernel", "nbformat": "nbformat",
}

def main():
    print("Python:", sys.version.split()[0], "|", platform.system(), platform.machine())
    print("Interpreter:", sys.executable)
    if sys.version_info[:2] != (3, 11):
        print("NOTE: Python 3.11 is the recommended project version.")
    failed = []
    for package, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            print(f"PASS import {module}: {metadata.version(package)}")
        except Exception as exc:
            failed.append(package)
            print(f"FAIL {package}: {type(exc).__name__}: {exc}")
    if failed:
        print("Setup incomplete:", ", ".join(failed))
        return 1
    import torch
    from torchvision.ops import nms
    # Verify matrix multiplication and autograd on CPU using synthetic values.
    x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    result = x @ torch.eye(2)
    torch.testing.assert_close(result, x)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    # Detect incompatible torch/torchvision binary builds.
    keep = nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([.9]), .5)
    assert keep.tolist() == [0]
    print("PASS CPU matrix multiplication, gradients, torchvision NMS")
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    if device != "cpu":
        t = torch.tensor([1., 2., 3.], device=device)
        torch.testing.assert_close((t * 2).cpu(), torch.tensor([2., 4., 6.]))
    print("PASS device operation:", device)
    print("ALL CHECKS PASSED. No dataset downloaded; no model trained.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
