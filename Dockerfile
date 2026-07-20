FROM maniskill/base

# Install PyTorch binaries directly from standard PyPI which bypasses the broken SSL handshake domains
RUN pip install --no-cache-dir \
    https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp39-cp39-linux_x86_64.whl

# Force a clean update of the main ManiSkill library dependencies
RUN pip install --no-cache-dir --upgrade mani_skill
