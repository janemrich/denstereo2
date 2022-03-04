FROM nvcr.io/nvidia/pytorch:21.10-py3

VOLUME denstereo

RUN pip install setproctitle==1.2.2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'@v0.6
RUN pip install mmcv==1.4.0
RUN pip install transforms3d==0.3.1
RUN pip install imageio==2.13.1
RUN pip install pypng==0.0.21
RUN pip install scikit-image==0.19.0
RUN pip install timm==0.4.12
RUN pip install plyfile==0.7.4
RUN pip install tensorboardX==2.4.1
