FROM  cosmosim

RUN apt-get update
RUN apt-get install -y \
     python3 python3-tk  \
     python3-pil python3-numpy python3-opencv \
     pybind11-dev libboost-python-dev
RUN apt-get install -y python3-pil.imagetk

WORKDIR /CosmoSim/Python/

RUN ln -s CosmoSim/CosmoSimPy.*.so CosmoSim/CosmoSimPy.so

RUN /usr/bin/python3 --version
RUN ls -l CosmoSim

RUN export uid=1002 gid=1002 && \
    mkdir -p /home/user && \
    echo "user:*:${uid}:${gid}:User,,,:/home/user:/bin/bash" >> /etc/passwd && \
    echo "user:*:${uid}:" >> /etc/group && \
    chown ${uid}:${gid} -R /home/user

USER user
CMD /usr/bin/python3 CosmoGUI.py
