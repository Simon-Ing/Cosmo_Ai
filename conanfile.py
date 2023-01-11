
from conans import ConanFile, CMake, tools
from conans.tools import OSInfo


class CosmoSimConan(ConanFile):
    name = "CosmoSim"
    author = "Lars Ivar Hatledal"
    license = "MIT"

    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    requires = (
        "symengine/0.9.0",
        "opencv/4.5.5",
        "zlib/1.2.13"
    )

    def requirements(self):
        info = OSInfo()
        if info.is_linux:
            self.requires("wayland/1.21.0")
