FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        sudo \
        wget \
	python3-pip \
	git \
	lsof \
	curl
RUN pip3 install tensorflow==2.4.1 tensorboard==2.4.1 numpy==1.19.2 PyYAML==5.3.1 jupyterlab pandas pyspark keras scikit-learn pytorch-lightning==1.3.0 ax-platform matplotlib pandas_datareader \
                 torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html	
WORKDIR /workspace
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''", "--no-browser"]
