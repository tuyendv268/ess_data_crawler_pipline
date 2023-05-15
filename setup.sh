apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
pip install pytorch-lightning==1.8
pip install torchmetrics==0.10.3