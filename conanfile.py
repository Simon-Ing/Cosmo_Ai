
from conan import ConanFile
from conan.tools.cmake import cmake_layout


class CosmoSimConan(ConanFile):
    name = "CosmoSim"
    license = "MIT"

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def configure(self):
        self.options["opencv"].with_ffmpeg = False

    def requirements(self):
        self.requires( "symengine/0.14.0" )
        self.requires( "opencv/4.13.0" )

    def layout(self):
        cmake_layout(self)

