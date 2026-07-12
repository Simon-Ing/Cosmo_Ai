#!/usr/bin/env python3
"""Patch the skbuild-conan conan profile for Windows wheel builds.

CosmoSim's C++ only uses ``cv::imread`` (OpenCV imgcodecs, i.e. PNG/JPEG), so
OpenCV's video I/O is dead weight.  OpenCV's conan recipe defaults
``with_ffmpeg=True``, which compiles ffmpeg + libaom-av1 from source on MSVC --
the slow, intermittently failing step behind the historically "erratic" Windows
wheel builds.  Disabling it removes that whole subtree; ``cv::imread`` keeps
working via the default ``with_png``/``with_jpeg`` backends.

skbuild-conan (v1.5.0) exposes no way to pass conan *options* through
``setup()`` (only requirements/settings), so we inject an ``[options]`` section
into the profile it uses.  skbuild-conan resolves the wheel build with the
``skbuild_conan_py`` profile and, if that profile already exists, reuses it
verbatim -- so the options we append here apply to the real dependency
resolution.  This script is invoked from ``[tool.cibuildwheel.windows]``
``before-all`` in ``pyproject.toml`` after ``conan profile detect``.

Using ``conan profile path`` (rather than a hard-coded ``~/.conan2`` path) keeps
this correct even if ``CONAN_HOME`` is overridden on the runner.
"""

import subprocess
import sys

PROFILE = "skbuild_conan_py"

# Only the load-bearing option: drop the ffmpeg/libaom-av1 source build.
OPTIONS = [
    "opencv/*:with_ffmpeg=False",
]


def main() -> int:
    path = subprocess.run(
        ["conan", "profile", "path", PROFILE],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    block = "\n[options]\n" + "\n".join(OPTIONS) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(block)

    print(f"Patched conan profile '{PROFILE}' at {path} with:{block}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
