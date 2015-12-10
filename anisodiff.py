import sys
import os.path
sys.path.append(os.path.join('', 'util'))
from timer import Timer

import numpy as np
import pylab
import warnings
 
''' 
This anisotropic diffusion serial (vectorized) implementation is adapted from 
the one published by the University of Oxford.
'''

def anisodiff_vec(img,niter=10,kappa=35,gamma=0.2,step=(1.,1.),option=1,ploton=False):
        """
        Anisotropic diffusion.
 
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
 
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration
 
        Returns:
                imgout   - diffused image.
 
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
 
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
 
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
 
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
 
        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
 
        Original MATLAB code by Peter Kovesi  
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>
 
        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>
 
        June 2000  original version.      
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
 
        # ...you could always diffuse each color channel independently if you
        # really want
        if img.ndim == 3:
                warnings.warn("Only grayscale images allowed, converting to 2D matrix")
                img = img.mean(2)
 
        # initialize output array
        img = img.astype('float64')
        imgout = img.copy()
 
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        deltaN = deltaS.copy()
        deltaW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
        gN = gS.copy()
        gW = gS.copy()
 
        # create the plot figure, if requested
        if ploton:
                import pylab as pl
                from time import sleep
 
                fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
                ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
                ax1.imshow(img,interpolation='nearest')
                ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
                ax1.set_title("Original image")
                ax2.set_title("Iteration 0")
 
                fig.canvas.draw()
 
        for ii in xrange(niter):
 
                # calculate the diffs of the four neighbors
                deltaS[:-1,: ] = np.diff(imgout,axis=0)
                deltaE[: ,:-1] = np.diff(imgout,axis=1)
                deltaN[1:, :] = -np.diff(imgout,axis=0)
                deltaW[: ,1:] = -np.diff(imgout,axis=1)

                # conduction gradients (only need to compute one per dim!)
                if option == 1:
                        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                        gE = np.exp(-(deltaE/kappa)**2.)/step[1]
                        gN = np.exp(-(deltaN/kappa)**2.)/step[0]
                        gW = np.exp(-(deltaW/kappa)**2.)/step[1]
                elif option == 2:
                        gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                        gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
                        gN = 1./(1.+(deltaN/kappa)**2.)/step[0]
                        gW = 1./(1.+(deltaW/kappa)**2.)/step[1]
                elif option == 3:
                        gS, gE, gN, gW = 1, 1, 1, 1
 
                # update matrices
                E = gE*deltaE
                S = gS*deltaS
                N = gN*deltaN
                W = gW*deltaW
                
                # update the image
                imgout += gamma/4*(E+S+N+W)

                if ploton:
                        iterstring = "Iteration %i" %(ii+1)
                        ih.set_data(imgout)
                        ax2.set_title(iterstring)
                        fig.canvas.draw()
                        # sleep(0.01)
 
        return imgout


if __name__ == '__main__':
    input_image = np.load('image.npz')['image'].astype(np.float32)
    pylab.gray()

    pylab.imshow(input_image)
    pylab.title('original image')

    pylab.figure()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    pylab.title('before - zoom')

    with Timer() as t:
        new_image = anisodiff_vec(input_image, 40)
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()

