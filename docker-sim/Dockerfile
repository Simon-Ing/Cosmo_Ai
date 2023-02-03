FROM  danger89/cmake:4.5

RUN apt-get update
RUN apt-get install -y libgtk2.0-dev libva-dev libx11-xcb-dev \
     libfontenc-dev libxaw7-dev libxkbfile-dev libxmuu-dev \
     libxpm-dev libxres-dev libxtst-dev libxvmc-dev \
     libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev \
     libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev \
     libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev \
     libxcb-xinerama0-dev libxcb-dri3-dev libxcb-util-dev \
     libxcb-util0-dev libvdpau-dev \
     libxss-dev libxxf86vm-dev \
     apt-utils \
     pybind11-dev build-essential

RUN pip3 install conan 
RUN conan profile new default --detect  # Generates default profile detecting GCC and sets old ABI
RUN conan profile update settings.compiler.libcxx=libstdc++11 default  # Sets libcxx to C++11 ABI

RUN git clone -b  develop https://github.com/CosmoAI-AES/CosmoSim.git
WORKDIR CosmoSim
RUN sh build.sh
