FROM  ubuntu:20.04

RUN apt-get update
RUN apt install -y \
          libgtk2.0-dev \
          libva-dev \
          libvdpau-dev \
          libx11-xcb-dev \
          libfontenc-dev \
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
          libxcb-util0-dev
RUN pip3 install conan==1.59

RUN conan profile new default --detect  
RUN conan profile update settings.compiler.libcxx=libstdc++11 default  

RUN git clone -b  develop https://github.com/CosmoAI-AES/CosmoSim.git
WORKDIR CosmoSim
RUN sh build.sh
RUN python3 Python/datagen.sh
