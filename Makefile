UV ?= uv
MPLCONFIGDIR ?= $(CURDIR)/.matplotlib
KERNEL_NAME ?= xallergen2
KERNEL_DISPLAY_NAME ?= Python (xallergen2)

.PHONY: lock sync kernel setup doctor clean

lock:
	$(UV) lock

sync:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(UV) sync --frozen

kernel: sync
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(UV) run python -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "$(KERNEL_DISPLAY_NAME)"

setup: sync

doctor:
	mkdir -p "$(MPLCONFIGDIR)"
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(UV) run python -c "import sys; import Bio, captum, huggingface_hub, ipykernel, matplotlib, numpy, pandas, requests, sklearn, seaborn, statsmodels, torch, tqdm, transformers; print(sys.executable); print(sys.version); print('biopython', Bio.__version__); print('ipykernel', ipykernel.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"

clean:
	rm -rf .venv
