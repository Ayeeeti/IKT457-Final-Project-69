# IKT457-Final-Project-69
Final project for Tsetlin machines
This is the final project 

## Install dependencies

Install the project's standard Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The CUDA training script (`vanilla_TM_cuda.py`) uses the
`PyHierarchicalTsetlinMachineCUDA` package. Install its dependencies separately
with:

```bash
python -m pip install -r requirements-cuda.txt
```

The CUDA backend also requires a working NVIDIA CUDA/PyCUDA setup. The regular
`requirements.txt` remains suitable for using the non-CUDA scripts.

On Windows, CUDA compilation also requires the Visual Studio C++ Build Tools
environment. Run the vanilla CUDA script through the provided launcher so that
environment is initialized:

```powershell
.\run_vanilla_TM_cuda.bat --epochs 1
```

The launcher uses the Python 3.14 environment. Omit `--epochs 1` to use the
script's default number of epochs.
