#@title STEP 1 复制代码库并安装运行环境
#@markdown #STEP 1 (6 min)
#@markdown ##复制代码库并安装运行环境
#@markdown ##Clone repository & Build environment

#git clone https://github.com/Plachtaa/VITS-fast-fine-tuning.git
pip install --upgrade --force-reinstall regex
pip install --force-reinstall soundfile
pip install --force-reinstall gradio==3.50.2
pip install imageio==2.4.1
pip install --upgrade youtube-dl
pip install moviepy
#cd VITS-fast-fine-tuning

python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --upgrade numpy
python -m pip install --upgrade --force-reinstall numba
python -m pip install --upgrade Cython

python -m pip install --upgrade pyzmq
python -m pip install pydantic==1.10.4
python -m pip install ruamel.yaml
python -m pip install git+https://github.com/openai/whisper.git
python -m pip install gdown

# build monotonic align
cd monotonic_align/
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
mkdir pretrained_models

