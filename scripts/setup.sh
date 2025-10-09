# Base server
conda env create -f env/environment-cu124.yml
conda activate sn17-gen-v1

# Trellis
cd scripts/trellis
bash setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# Back to root
cd ../..
pip install -r requirements.txt
