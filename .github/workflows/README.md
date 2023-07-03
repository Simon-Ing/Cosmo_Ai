The following workflows are defined

+ On push to develop or master: `trial-run.yml`
    + This builds the system for Windows and Linux, using one configuration for each.
    + It also verifies that the CLI tool runs without parameters.
+ The release workflow: `release.yml`
    + This is run on a push to a tag prefixed by `v`; e.g. `git push origin v0.1.1`
    + It creates all artifacts and uploads them to the release.
+ Manually triggered workflows 
    + `comprehensive-test.yml` runs the regression test on several builds.
        + It currently works on Linux.
        + It fails on Windows because Sobel is not defined in simaux.cpp
    + `build-all.yml` builds on several configurations on Linux and Windows
    + `darwin.yml` builds for MacOS but does not presently work.
    + `test.yml` is a simple test used when learning workflows.  Not to be used.
