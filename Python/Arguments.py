
import argparse

class CosmoParser(argparse.ArgumentParser):
  def __init__(self,*a,**kw):
    super().__init__(*a,**kw)
    self.add_argument('-x', '--x', default=0, help="x coordinate")
    self.add_argument('-y', '--y', default=0, help="y coordinate")
    self.add_argument('-T', '--phi', help="polar coordinate angle (phi)")

    self.add_argument('-s', '--sigma', default=20, help="source size (sigma)")
    self.add_argument('-2', '--sigma2', default=10, help="secondary source size (sigma2)")
    self.add_argument('-t', '--theta', default=45, help="source rotation angle (theta)")

    self.add_argument('-X', '--chi', default=50, help="lens distance ration (chi)")
    self.add_argument('-E', '--einsteinradius', default=20, help="Einstein radius")

    self.add_argument('-n', '--nterms', default=10, help="Number of Roulettes terms")
    self.add_argument('-Z', '--imagesize', default=400, help="image size")

    self.add_argument('-l', '--lensmode',
            default="SIS", help="lens mode")
    self.add_argument('-L', '--modelmode',
            default="Point Mass (exact)", help="lens mode")
    self.add_argument('-S', '--sourcemode',
            default="Spherical", help="source mode")

    self.add_argument('-R', '--reflines',action='store_true',
            help="Add reference (axes) lines")
    self.add_argument('-C', '--centred',action='store_true', help="centre image")
    self.add_argument('-M', '--mask',action='store_true',
            help="Mask out the convergence circle")
    self.add_argument('-m', '--showmask',action='store_true',
            help="Mark the convergence circle")

    self.add_argument('-N', '--name', default="test",
            help="simulation name")
    self.add_argument('-D', '--directory',default="./",
            help="directory path (for output files)")
    self.add_argument('-O', '--maskscale',default="0.9",
            help="Scaling factor for the mask radius")
    self.add_argument('-c', '--components',default="6",
            help="Number of components for joined image")

    self.add_argument('-P', '--psiplot',action='store_true',default=False,
            help="Plot lens potential as 3D surface")
    self.add_argument('-K', '--kappaplot',action='store_true',default=False,
            help="Plot mass distribution as 3D surface")

    self.add_argument('-f', '--family',action='store_true',
            help="Several images moving the viewpoint")
    self.add_argument('-J', '--join',action='store_true',
            help="Join several images from different viewpoints")
    self.add_argument('-F', '--amplitudes',help="Amplitudes file")
    self.add_argument('-A', '--apparent',action='store_true',help="write apparent image")
    self.add_argument('-a', '--actual',action='store_true',help="write actual image")
    self.add_argument('-U', '--original',action='store_true',help="write original image before centring")
    self.add_argument('-o', '--outfile',
            help="Output CSV file")
    self.add_argument('-i', '--csvfile',
            help="Dataset to generate (CSV file)")
