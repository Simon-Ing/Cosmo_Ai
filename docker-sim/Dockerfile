FROM  ubuntu:20.04

# The following line is needed to prevent docker build from halting
# on time zone selection on newer versions of docker.
ENV DEBIAN_FRONTEND=nonintercative

RUN apt-get update
RUN apt-get install -y \
          libva-dev \
          libvdpau-dev \
          libx11-xcb-dev \
          libfontenc-dev  \
          libxaw7-dev \libxkbfile-dev \
          libxmu-dev \
          libxmuu-dev \
          libxpm-dev \
          libxres-dev \
          libxss-dev \
          libxtst-dev \
          libxv-dev \
          libxvmc-dev \
          libxxf86vm-dev \
          libxcb-render-util0-dev \
          libxcb-xkb-dev \
          libxcb-icccm4-dev \
          libxcb-image0-dev \
          libxcb-keysyms1-dev \
          libxcb-randr0-dev \
          libxcb-shape0-dev \
          libxcb-sync-dev \
          libxcb-xfixes0-dev \
          libxcb-xinerama0-dev \
          libxcb-dri3-dev \
          libxcb-util-dev \
          libxcb-util0-dev \
          libgtk2.0-dev 
RUN apt-get install -y \
     python3 python3-tk  \
     python3-pil python3-numpy python3-opencv \
     pybind11-dev libboost-python-dev \
     python3-pil.imagetk \
     python3-pip
RUN pip3 install conan==1.59

RUN conan profile new default --detect  
RUN conan profile update settings.compiler.libcxx=libstdc++11 default  

RUN apt-get install -y \
    build-essential git cmake
RUN git clone -b  develop https://github.com/CosmoAI-AES/CosmoSim.git
WORKDIR CosmoSim
RUN sh build.sh


RUN /usr/bin/python3 --version

WORKDIR /CosmoSim/CosmoSimPy/
USER user
CMD /usr/bin/python3 CosmoGUI.py

# RUN python3 Python/datagen.sh
