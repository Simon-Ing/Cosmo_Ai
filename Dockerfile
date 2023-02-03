FROM  cosmosim

RUN apt-get update
RUN apt-get install -y \
     python3 python3-tk  \
     pybind11-dev libboost-python-dev

WORKDIR /CosmoSim/Python/

RUN ln -s CosmoSim/CosmoSimPy.*.so CosmoSim/CosmoSimPy.so

RUN /usr/bin/python3 --version
RUN ls -l CosmoSim

CMD /usr/bin/python3 CosmoGUI.py
