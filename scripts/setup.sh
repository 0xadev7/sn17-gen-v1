# Base server
conda env create -f env/environment-cu124.yml
conda activate sn17-miner-v1

# Trellis
cd trellis
bash setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
