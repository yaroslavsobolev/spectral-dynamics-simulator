# -*- coding: utf-8 -*-
"""
Single molecule spectral dynamics simulator. Version 0.1.

A module for simulating the spectral trails of a single molecule given the
certain elementary excitations in its local neighbourhood.

Local neighbourhood
is (for now) made up of only the two-level systems (TLSs) whose parameters are
distributed in a user-specified way. By default, these distributions are the
ones used in (Geva, Skinner, 1997) paper. The default constants are the ones
for system of tetra-tertbutylterrylene(TBT) in amorphous polyisobutylene (PIB).
Molecular weight of PIB is considered to be around 420 000 g/mol.

Created on Thu Jun  5 15:58:00 2014

@author: Yaroslav Igorevich Sobolev
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline
import math
import time
import pickle

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical
    distribution. Algo from http://stackoverflow.com/questions/5408276/
    python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0, np.pi*2)
    costheta = np.random.uniform(-1, 1)    
    theta = np.arccos(costheta)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return (x, y, z)

#def nonlinspace(y, xbegin, xend, size):
#    ysize = len(y) 
#    default_spacing = (xend-xbegin)/size
#    y_deriv = np.diff(y, n=1)
#    np.append(y_deriv, 0)
#    y_deriv2 = np.diff(y, n=2)
#    np.append(y_deriv2, 0)
    
def OptimalBins(x):        
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    bin_width = iqr*2/(len(x)**(1/3.0)) 
    bins = np.arange(min(x), max(x) + bin_width, bin_width)
    return bins

#==============================================================================
# Distrobution of distances of TLSs from central chromophore
#==============================================================================
# The distribution here is such that density increases as R^2, to
# mimic the density of points in spherical layer of radius R when
# points are distributed in 3D space evenly
class TLS_r_gen(stats.rv_continuous):
    '''Distrobution of distances of TLSs from central chromophore'''
#    def __init__(self, Rmin, Rmax):
#        super().__init__(a=Rmin, b=Rmax)    
    
    @classmethod
    def _cdf(self, x, Rmin, Rmax):
        ro = 1 / (4/3*np.pi*Rmax**3 - 4/3*np.pi*Rmin**3)
        res = ro*(4/3*np.pi*x**3 - 4/3*np.pi*Rmin**3)
        return res
    @classmethod
    def _ppf(self, x, Rmin, Rmax):
        ro = 1 / (4/3*np.pi*Rmax**3 - 4/3*np.pi*Rmin**3)
        return (3/4/np.pi * (x/ro + 4/3*np.pi*(Rmin**3)))**(1/3)

    @classmethod
    def _pdf(self, x, Rmin, Rmax):
        ro = 1 / (4/3*np.pi*Rmax**3 - 4/3*np.pi*Rmin**3)
        return ro*4*np.pi*x**2
        
    @classmethod
    def _stats(self):
        return 0., 0., 0., 0.

#==============================================================================
# Distrobution of potential asymmetries (A)
#==============================================================================
class TLS_A_gen(stats.rv_continuous):
    '''Distrobution of potential asymmetries (A)'''

#    def __init__(self, Amax):
#        super().__init__(a=0, b=Amax)   
        
    @classmethod
    def _cdf(self, x, mu, Amax):
        return (x/Amax)**(1+mu)

    @classmethod
    def _ppf(self, x, mu, Amax):
        return Amax*(x**(1/(1+mu)))
        
    @classmethod
    def _pdf(self, x, mu, Amax):
        return (1+mu)/(Amax**(1+mu)) * (x**mu)

    @classmethod
    def _stats(self):
        return 0., 0., 0., 0.

#==============================================================================
# Distrobution of potential barriers (J)
#==============================================================================
class TLS_J_gen(stats.rv_continuous):
    '''Distrobution of potential barriers (J)'''

#    def __init__(self, Jmin, Jmax):
#        super().__init__(a=Jmin, b=Jmax)  

    @classmethod
    def _cdf(self, x, Jmin, Jmax):
        return np.log(x/Jmin)/np.log(Jmax/Jmin)

    @classmethod
    def _ppf(self, x, Jmin, Jmax):
        return Jmin*((Jmax/Jmin)**x)

    @classmethod
    def _pdf(self, x, Jmin, Jmax):
        return 1/np.log(Jmax/Jmin)*(1/x)

    @classmethod
    def _stats(self):
        return 0., 0., 0., 0.


class VirtualLocalEnvironment:
    """A class that creates and stores the local environment of the molecule"""

    def distribute_TLS_parameters(self, PlottingOn = False, do_cutting = False,
                                  volume_plotting = False,
                                  distance_to_surface = 10, 
                                  number_of_TLS_on_surface = 0):
        """Makes a set of TLSs using the distributions with given parameters"""
        
        if not do_cutting:
            number_of_TLS_on_surface = 0
            
        # Just for plotting in 3D, not useful for simulations
        if volume_plotting:
            xx = []
            yy = []
            zz = []
        
        # First distribute the usual TLSs uniformly, with usual Rs
        Gener = TLS_r_gen(name='TLS_r_gen')
        Gener.a = self.Rmin
        Gener.b = self.Rmax
        self.TLS_r = Gener.rvs(size=self.number_of_TLS - \
                                    number_of_TLS_on_surface,
                               Rmin=self.Rmin,
                               Rmax=self.Rmax)
                               
        # Now add number_of_TLS_on_surface number_of_TLS_on_surface TLSs
        # in the circle located distance_to_surface nanometers from
        # the (0,0,0) point.
        if do_cutting:
            while len(self.TLS_r) < self.number_of_TLS:
                x, y = np.random.uniform(low = -30, high = 30, size = 2)
                if (x**2 + y**2) < (30**2 - distance_to_surface**2) and \
                    (math.sqrt(x**2 + y**2 + distance_to_surface**2) > self.Rmin):
                    distance_from_origin = math.sqrt(x**2 + y**2 + \
                                                     distance_to_surface**2)
                    self.TLS_r = np.append(self.TLS_r, distance_from_origin)
                    if volume_plotting:
                        xx.append(x)
                        yy.append(y)
                        zz.append(distance_to_surface)
            
            
            
        # Draw a sample
        Gener = TLS_A_gen(name='TLS_A_gen')
        Gener.a = 0
        Gener.b = self.Amax
        self.TLS_A = Gener.rvs(size=self.number_of_TLS, Amax=self.Amax,
                               mu=self.mu)

        # Draw a sample
        Gener = TLS_J_gen(name='TLS_J_gen')
        Gener.a = self.Jmin
        Gener.b = self.Jmax
        self.TLS_J = Gener.rvs(size=self.number_of_TLS,
                               Jmin=self.Jmin, Jmax=self.Jmax)

        self.TLS_epsilon = 2*np.random.randint(2, size=self.number_of_TLS)-1

        #compute the remaining parameters
        # NOT OPTIMIZED HERE
        self.TLS_E = (self.TLS_J**2 + self.TLS_A**2)**(0.5)

        self.TLS_Kdown = self.tls_phonon_constant*(self.TLS_J**2) * \
                self.TLS_E* \
                1 / (1 - np.exp(float(-1/self.temperature)*self.TLS_E))

        self.TLS_Kup = self.tls_phonon_constant*(self.TLS_J**2) * \
                self.TLS_E* \
                1 / (np.exp(float(1/self.temperature)*self.TLS_E) - 1)

#        self.TLS_Kup = np.array([0.1])
#        self.TLS_Kdown = np.array([1])

        self.TLS_K = self.TLS_Kdown + self.TLS_Kup
        
        # Cutting out some parts of the sphere
        if do_cutting:
            list_of_TLS_indices_for_deletion = []
            for i in range(self.number_of_TLS - number_of_TLS_on_surface):
                x,y,z = random_three_vector()
                if z*self.TLS_r[i] > distance_to_surface:
                    list_of_TLS_indices_for_deletion.append(i)
                elif volume_plotting:
                    xx.append(x*self.TLS_r[i])
                    yy.append(y*self.TLS_r[i])
                    zz.append(z*self.TLS_r[i])
            
            self.TLS_r = np.delete(self.TLS_r, list_of_TLS_indices_for_deletion)        
            self.TLS_A = np.delete(self.TLS_A, list_of_TLS_indices_for_deletion)
            self.TLS_J = np.delete(self.TLS_J, list_of_TLS_indices_for_deletion)
            self.TLS_epsilon = np.delete(self.TLS_epsilon, 
                                         list_of_TLS_indices_for_deletion)
            self.TLS_E = np.delete(self.TLS_E, list_of_TLS_indices_for_deletion)
            self.TLS_Kdown = np.delete(self.TLS_Kdown, 
                                       list_of_TLS_indices_for_deletion)
            self.TLS_Kup = np.delete(self.TLS_Kup, 
                                     list_of_TLS_indices_for_deletion)
            self.TLS_K = np.delete(self.TLS_K, list_of_TLS_indices_for_deletion)
            self.number_of_TLS = len(self.TLS_r)
            print('{0} TLSs remained after cutting'.format(self.number_of_TLS))     
        
        #Plotting in 3D
        if volume_plotting:
            import pylab
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = pylab.figure()
            ax = Axes3D(fig)
            
            ax.scatter(xx, yy, zz)
            plt.show()        
            
        if PlottingOn:
            self.plot_hstograms_for_TLS_parameters()

    def plot_hstograms_for_TLS_parameters(self, NumberOfBins=50):
        f = plt.figure(1)
        f.add_subplot(311)
        plt.hist(self.TLS_r, bins=NumberOfBins, alpha=0.3)
        f.add_subplot(312)
        plt.hist(self.TLS_A, bins=NumberOfBins, alpha=0.3)
        ax = f.add_subplot(313)
        ax.set_yscale('log')
        plt.hist(self.TLS_J, bins=NumberOfBins, alpha=0.3)
        f.show()

    def __init__(self,
                 ro_p = 3e-4,       # calibration spatio-energetic 
                                    #                       density of TLSs
                 #number_of_TLS = 10, #number of TLSs, NOW DEFINED BY DENSITY
                 alpha = 25,          # TLS-chromopnore coupling constant
                 Rmax = 30,           # Максимальное расстояние до молекулы,
                                      # в нанометрах
                 Rmin = 2.5,
                 Amax = 40,           # Максимальная асимметрия потенциала
                 mu = 0.3,            # Показатель в распределении ассиметрий
                                      # потенциалов
                 Jmin = 0.000000276,   # Минимальный туннельный элемент
                 Jmax = 40,           # Максимальный туннельный элемент
                 tls_phonon_constant = 1.16e9, #Константа ДУС-фонон
                 temperature = 4.5,    # Текущая температура
                 testing=False,          # режим строго заданных ДУС
                 do_cutting=False,
                 distance_to_surface=0,
                 surface_TLS_density = 2 ## TLSs per square nm
                 ):
        # Set the constants.

        #    ~ "In basement of Tautology Club, police has found a
        #          fatally murdered corpse of a deceased deadman" ~

        #self.number_of_TLS = number_of_TLS  #number of TLSs
        self.alpha = alpha          # TLS-chromopnore coupling constant
        self.Rmax = Rmax            # Максимальное расстояние до молекулы,
                                    # в нанометрах
        self.Rmin = Rmin
        self.Amax = Amax            # Максимальная асимметрия потенциала
        self.mu = mu                # Показатель в распределении ассиметрий
                                    #                        потенциалов
        self.Jmin = Jmin            # Минимальный туннельный элемент
        self.Jmax = Jmax            # Максимальный туннельный элемент
        self.tls_phonon_constant = tls_phonon_constant #Константа ДУС-фонон
        self.temperature = temperature
        
        # This is from Appendix B of (Geva, Skinner, 1997)
        # It deals with calculating spatial density of TLSs
        #  for use in subsequent simulations.
        #
        # This density should be dependent on the choice of Amax, Jmin, Jmax
        #  in some way - this much is obvious. This section deals with explicit
        #  calibration of these matters.
        P_0 = (1+mu) / ((Amax**(1+mu))*np.log(Jmax/Jmin))
        
        E1 = 0.5
        E2 = 1.5
        a = 1.4485/Jmin 
        B7 = 9/16 * E2**(4/3) * (4/3*np.log(a*E2) - 1) - \
             9/16 * E1**(4/3) * (4/3*np.log(a*E1) - 1)
        #print('Factor is {0}'.format(P_0*B7))
        
        ro = ro_p / (P_0*B7)
        self.number_of_TLS = int(round(ro * (4/3*np.pi*Rmax**3 - \
                                                        4/3*np.pi*Rmin**3)))
        print('TLS density is {0:.4f} nm^-3; '\
              'there are {1} TLSs in {2}-nm-radius sphere.'\
            .format(ro, self.number_of_TLS, Rmax))
        
        # Calculate the number of TLSs on surface
        #   First, get the surface area. It's a circle            
        if (distance_to_surface >= Rmax) or (not do_cutting):
            no_surface_please = True
            surface_area = 0
            number_of_TLS_on_surface = 0
            do_cutting = False
            print('Surface is too far.')
        else:
            no_surface_please = False
            surface_area = np.pi*(30**2 - distance_to_surface**2)
            number_of_TLS_on_surface = int(surface_TLS_density*surface_area)
            print('Surface area {0:.2f} nm^2, '
                  'there are {1} TLSs on surface.'.format(
                      surface_area,
                      number_of_TLS_on_surface))
                  
            self.number_of_TLS = self.number_of_TLS + number_of_TLS_on_surface
        
        if not testing:
        # Do the routine of distributing TLS parameters
            if not no_surface_please:
                self.distribute_TLS_parameters(
                    do_cutting = do_cutting,
                    distance_to_surface = distance_to_surface,
                    number_of_TLS_on_surface = number_of_TLS_on_surface)
            else:
                self.distribute_TLS_parameters(
                    do_cutting = False,
                    number_of_TLS_on_surface = 0)                
            
        else:
            self.number_of_TLS = 0

            self.TLS_r = np.array([5.3353456088119628])
            self.TLS_J = np.array([1.9654667275946586e-05])
            self.TLS_epsilon = np.array([-1])
            self.TLS_A = np.array([0.013223931288874011])
            
#            self.TLS_r = np.array([5.3353456088119628, 4])
#            self.TLS_J = np.array([1.9654667275946586e-05, 2e-05])
#            self.TLS_epsilon = np.array([-1, 1])
#            self.TLS_A = np.array([0.013223931288874011, 4])
            
            
            self.TLS_E = (self.TLS_J**2 + self.TLS_A**2)**(0.5)
    
            self.TLS_Kdown = self.tls_phonon_constant*(self.TLS_J**2) * \
                    self.TLS_E* \
                    1 / (1 - np.exp(float(-1/self.temperature)*self.TLS_E))
    
            self.TLS_Kup = self.tls_phonon_constant*(self.TLS_J**2) * \
                    self.TLS_E* \
                    1 / (np.exp(float(1/self.temperature)*self.TLS_E) - 1)
                    
            self.TLS_nu = 2*math.pi*self.alpha*\
                    self.TLS_A/self.TLS_E*\
                    self.TLS_epsilon/(self.TLS_r)**3

            self.TLS_K = self.TLS_Kdown + self.TLS_Kup            
                
            print('{0} TLSs, K = {1}, p = {2}, nu/2pi = {3}'.format(
                                                    self.number_of_TLS,
                                                    self.TLS_K,
                                                    self.TLS_Kup/self.TLS_K,
                                                    self.TLS_nu/2/np.pi
                                                       ))
    #        self.TLS_Kup = np.array([0.1])
    #        self.TLS_Kdown = np.array([1])
                
class MoleculeSpectrumSimulator:
    
    def __init__(self, max_time = 3750, steps_per_sec = 6.45,
                 step_subdivisor = 10, spectral_range = 30,
                 steps_in_scan = 500,
                 do_cutting=False,
                 distance_to_surface=10):
        self.instance_id = 1
        self.max_time = max_time        #:double ; in SECONDS!!!!
        self.steps_per_sec = steps_per_sec    #:integer ; #Should have a default,
                                            # but be actively used by user
        self.steps_in_scan = steps_in_scan
        self.secs_per_step = 1/steps_per_sec

        # StepsPerBitmap # WEIRD VARIABLE #:integer ; #Should have a default;
        self.step_subdivisor = step_subdivisor  #:integer, Should have a default
        self.spectral_range = spectral_range

        self.VLE = VirtualLocalEnvironment(testing = False,
                                           do_cutting = do_cutting,
                                           distance_to_surface = 
                                               distance_to_surface)
        self.number_of_TLS = self.VLE.number_of_TLS
        
        self.T1 = 1/(2*np.pi*42e6)
        self.total_microsteps = max_time*steps_per_sec*step_subdivisor
        
        #self.v = {0.00:0.00}
        self.threshold_between_fast_and_slow = 50/(self.secs_per_step*self.steps_in_scan)
        #self.threshold_between_fast_and_slow = 1/10

        
        self.sides_multiplier = 2 # This is a value that makes extra ranges
                        # to the left and to the right of the main spectral
                        # range of fast-brodened spectrum function.
                        # This is to deal with broadening that is larger than
                        # the spectral range and simulate this
                        # broadening properly.
                        # Value 2 means that there are 2 30-GHz-ranges to the
                        # left and 2 3-GHz-ranges to the right - total of 
                        # 5 ranges per entire span of fast-broadened functiion
        
        #Increased precision when calculating the fast-broadened spectrum.
        #   This value is chosen so that the TLSs with splittings less than
        #   1 MHz / increase_precision would give total broadening no more
        #   than maximum_erroneous_broadening MHz. This seems reasonable.
        maximum_erroneous_broadening = 100 # MHz
        self.increase_precision = self.number_of_TLS/maximum_erroneous_broadening
        #self.increase_precision = 1
        print("Increase precision = {0:.2f}".format(self.increase_precision))

    def subdivide_TLSes_by_speeds(self):
        # Find such TLSes that their 1/kUP and 1/kDOWN are very small - less than
        # 1/100 of the time microstep and mark them as superfast
        self.fast_TLSes = [k for k in range(self.number_of_TLS)
                     if min(self.VLE.TLS_Kup[k],self.VLE.TLS_Kdown[k]) > \
                     self.threshold_between_fast_and_slow
                    ]
        self.slow_TLSes = [k for k in range(self.number_of_TLS)
                     if min(self.VLE.TLS_Kup[k],self.VLE.TLS_Kdown[k]) <= \
                     self.threshold_between_fast_and_slow
                    ]
        self.number_of_slow_TLS = len(self.slow_TLSes)
        print("Number of slow TLSes is {0}".format(self.number_of_slow_TLS))
        self.number_of_fast_TLS = len(self.fast_TLSes)
        print("Number of fast TLSes is {0}".format(self.number_of_fast_TLS))

    def calculate_initial_fast_broadened_spectrum_naively(self, Plotting = False, 
                                              DoSubsample = True,
                                              verbose = False,
                                              Normalize = True):
        """Don't use this! It makes line widths twice larger than the correct
        values. Probably screws up other aspects of line shape as well."""
        
        if verbose:
            print('Calculating spectrum broadened by fast TLSs...')
        # create 1D array where indices are steps whose size is
        #                                  (1 MHz)/self.increase_precision
        self.base_spectrum = np.zeros(int(
            self.spectral_range* \
            (self.sides_multiplier*2 + 1)*1000* \
            self.increase_precision)
        )
                                         
        # I define original natural (lifetime-limited) linewidth as
        # something much less than 1 MHz/sides_multiplier/increase_precision
        self.base_spectrum[int(len(self.base_spectrum))//2] = 1
        
        x = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                         (1+self.sides_multiplier)*self.spectral_range,
                         len(self.base_spectrum))
                       
        ZeroSplittingsCounter = 0
        for k in self.fast_TLSes:
            if verbose:
                if k%50 == 0:
                    print('{0}% of fast TLSs are done.'\
                    .format(round(100*k/(self.number_of_TLS))))
            
            # Determine the spectral splitting caused by this TLS
            nu = self.VLE.alpha*self.VLE.TLS_A[k]/self.VLE.TLS_E[k]*\
                 self.VLE.TLS_epsilon[k]/(self.VLE.TLS_r[k])**3
            splitting_in_MHz = int(abs(nu*1000*self.increase_precision))
            if splitting_in_MHz == 0:
                #weird TLS that induces zero splitting
                ZeroSplittingsCounter += 1            
                continue
            else:
                
#==============================================================================
#        OLD IMPLEMENTATION - was too slow
#                 
#                 # Splitting funtion consists of two Dirac's delta-functions,
#                 # one delta-function at 0 and another at the spli-
#                 # tting caused by this TLS to the molecular spectrum
#                 # Magnitudes of these twi delta-functions are set to occupation
#                 # probabilities of upper and lower state of the TLS.
#                 SplitterFunction = np.zeros(splitting_in_MHz + 1)
#                 UpperStateOccupation = self.VLE.TLS_Kup[k]/self.VLE.TLS_K[k]
#                 LowerStateOccupation = self.VLE.TLS_Kdown[k]/self.VLE.TLS_K[k]
#                 if nu>0:
#                     SplitterFunction[0] = LowerStateOccupation
#                     SplitterFunction[-1] = UpperStateOccupation
#                 else:
#                     SplitterFunction[-1] = LowerStateOccupation
#                     SplitterFunction[0] = UpperStateOccupation
# 
#                 # Now convolve the current form of the spectrum with his new
#                 # "splitting function" that consists of two delta-functions.
#                 # This convolution is equivalent to adding the original spec-
#                 # trum to its copy shifted by the amount of MHz equal to
#                 # spectral shift induced by this TLS
#                 self.base_spectrum = np.convolve(self.base_spectrum,
#                                             SplitterFunction,
#                                             'same')
#==============================================================================
                
                UpperStateOccupation = self.VLE.TLS_Kup[k]/self.VLE.TLS_K[k]
                LowerStateOccupation = self.VLE.TLS_Kdown[k]/self.VLE.TLS_K[k]
                
                if nu > 0:
                    shifted_shadow = np.roll(self.base_spectrum,
                                             splitting_in_MHz)
                    shifted_shadow[:splitting_in_MHz] = 0
                else:
                    shifted_shadow = np.roll(self.base_spectrum,
                                             -1*splitting_in_MHz)
                    shifted_shadow[-1*splitting_in_MHz:] = 0
                    
                self.base_spectrum = \
                            np.add(LowerStateOccupation*self.base_spectrum, 
                                   UpperStateOccupation*shifted_shadow)
                                   
                if UpperStateOccupation < 0.4:
                    print('TLS id = {0}'.format(k))                   
                    print('Occupations: U {0:.2f}, D {1:.2f}' \
                                                     .format(UpperStateOccupation,
                                                             LowerStateOccupation))
                    print('Splitting = {0:.5f}'.format(nu))
#                plt.plot(x, self.base_spectrum)    
#                plt.show()
                
        
        if verbose:
            print('TLSs that induced zero splitting: {0}'.\
                                        format(ZeroSplittingsCounter))
        if Normalize:
            self.base_spectrum = self.base_spectrum / \
                                 np.linalg.norm(self.base_spectrum)
        else:
            self.base_spectrum = self.base_spectrum*1e8


        if Plotting:
            f = plt.figure(1)
            x = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                             (1+self.sides_multiplier)*self.spectral_range,
                             len(self.base_spectrum))
                            
            plt.plot(x, self.base_spectrum)
        
        #subsample for storage
        
        # find the full width at half maximum
        maxsignal = max(self.base_spectrum)
        x1 = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                         (1+self.sides_multiplier)*self.spectral_range,
                         len(self.base_spectrum))
                                    
        spl = UnivariateSpline(x1, self.base_spectrum - maxsignal/2, s=0)
        roots = spl.roots()
        print('FWHM intersection points are ' + str(roots))
        fwhm = max(roots) - min(roots)
        
        if DoSubsample:
            x1 = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                                     (1+self.sides_multiplier)*self.spectral_range,
                                     len(self.base_spectrum)
                                    )
            x2 = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                             (1+self.sides_multiplier)*self.spectral_range,
                             500
                            )
            spl = UnivariateSpline(x1, self.base_spectrum, s=0)
            self.base_spectrum = spl(x2)
        
        if Plotting:
            x = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                             (1+self.sides_multiplier)*self.spectral_range,
                             len(self.base_spectrum)
                            )        
            plt.plot(x, self.base_spectrum, 'o')
    
        x1 = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                         (1+self.sides_multiplier)*self.spectral_range,
                         len(self.base_spectrum)
                        )
        self.broadened_spectrum = UnivariateSpline(x1, self.base_spectrum, s=0)
        
        if Plotting:
            x = np.linspace(-1*self.sides_multiplier*self.spectral_range,
                             (1+self.sides_multiplier)*self.spectral_range,
                             30000
                            )        
            plt.plot(x, self.broadened_spectrum(x), color = 'r')
            f.show()
            
        return fwhm
#    def broadened_spectrum(self, v):
#        return np.interp(v, 
#                         np.linspace(0,
#                                     self.spectral_range, #in GHz
#                                     len(self.base_spectrum)
#                                    ),
#                         self.base_spectrum
#                        )
        
    def calculate_initial_fast_broadened_spectrum(self, plotting = False, 
                                                  verbose = True):
        from scipy.fftpack import fft
        from scipy.fftpack import fftshift
    
        def err_handler(type, flag):
            #print("Floating point error (%s), with flag %s" % (type, flag))
            a = 1+1
        saved_handler = np.seterrcall(err_handler)
        save_err = np.seterr(all='call')        
        
        minimal_step = 0.5/(self.spectral_range*1e9) # resolution is 10 MHz
        max_autocorr_time = 1*(self.T1)
        autocorr_length = int(max_autocorr_time/minimal_step)
        autocorr = np.ones(autocorr_length, dtype = np.complex128)
        t = np.linspace(0, max_autocorr_time, num = autocorr_length)
        minimal_step = t[1] - t[0]
        max_autocorr_time = t[-1]
        for k in self.fast_TLSes:
            if verbose:
                if k%5000 == 0:
                    print('{0}: {1}% of fast TLSs are done.'\
                    .format(
                        self.instance_id,
                        round(100*k/(self.number_of_TLS))))
            
            # Determine the spectral splitting caused by this TLS
            nu = 1e9*2*np.pi*self.VLE.alpha*self.VLE.TLS_A[k]/self.VLE.TLS_E[k]*\
                 self.VLE.TLS_epsilon[k]/((self.VLE.TLS_r[k])**3)
                 
            # Get the upper state population p and flip rate K (for brevity)
            p = self.VLE.TLS_Kup[k]/self.VLE.TLS_K[k]
            K = self.VLE.TLS_K[k]
            
            # Get alpha an omega. The Beginning and the End, muahaha!
            omega = np.sqrt(K*K/4 - nu*nu/4 - 1j*(p-1/2)*nu*K)            
            while omega > 1e8:
                K = K/10                
                omega = np.sqrt(K*K/4 - nu*nu/4 - 1j*(p-1/2)*nu*K)                
                
            alpha = K/2 - 1j*(p-1/2)*nu
            mult = np.cosh(omega*t) + alpha/omega*np.sinh(omega*t)
            autocorr = autocorr * np.exp(-1/2*(K+1j*nu)*t) * mult
        
        under_integral_term = autocorr*np.exp(-1*t/(2*self.T1))
#        if plotting:
#            plt.plot(t, 
#                     under_integral_term) #[0:autocorr_length/2]
#            plt.show()
      
        yf = 2.0/autocorr_length*np.abs(np.real(fft(under_integral_term)))        
        xf = (np.fft.fftfreq(autocorr_length, d=minimal_step)+15e9)/1e9
        
        xf = fftshift(xf)
        yf = fftshift(yf)
        
        if plotting:
            plt.ioff()
            plt.plot(xf, yf)
            plt.savefig('{0:06d}.png'.format(
                                    np.random.randint(100000, size=1)[0]
                                              ), 
                        dpi=100, 
                        bbox_inches = 'tight'
                        )
            plt.close()
            
        self.broadened_spectrum = UnivariateSpline(xf, yf, s=0)
        
        # find the full width at half maximum
#        x1 = np.linspace(-1*self.sides_multiplier*self.spectral_range,
#                         (1+self.sides_multiplier)*self.spectral_range,
#                         len(yf))
#        sig = self.broadened_spectrum(x1)                            
        maxsignal = max(yf)
        spl = UnivariateSpline(xf, yf - maxsignal/2, s=0)
        roots = spl.roots()
        print('FWHM intersection points are ' + str(roots))
        if roots.any():
            fwhm = max(roots) - min(roots)
        else:
            fwhm = 0
        
#        #plot
#        if plotting:
#            x = np.linspace(-1*self.sides_multiplier*self.spectral_range,
#                             (1+self.sides_multiplier)*self.spectral_range,
#                             30000
#                            )        
#            plt.plot(x, self.broadened_spectrum(x), color = 'r')
#            plt.show()
        
        return fwhm        
            
        
    # Generate initial state vector for all TLSs
    def generate_initial_states_for_TLSs(self):
        self.IsInUpperState = np.zeros(self.number_of_TLS, dtype = bool)
        for i in self.slow_TLSes:
            if np.random.uniform() < self.VLE.TLS_Kup[i]/self.VLE.TLS_K[i]:
                self.IsInUpperState[i] = True

    def generate_delays_sequence_for_slow_TLSs(self):
        """NEVER USER THIS METHOD, AS THERE IS NEVER ENOUGH RAM FOR IT"""
        self.TLS_delays_up = tuple([np.random.exponential(scale = 1/K,
                                                   size = self.max_time/(1/K))
                                for K in self.VLE.TLS_Kup[self.slow_TLSes]])

        self.TLS_delays_down = tuple([np.random.exponential(scale = 1/K,
                                                   size = self.max_time/(1/K))
                                for K in self.VLE.TLS_Kdown[self.slow_TLSes]])

        # interleave two arrays (up and down) for each moe
        self.TLS_delays = [np.empty((self.TLS_delays_up[i].size +
                                        self.TLS_delays_down[i].size,),
                                          dtype=self.TLS_delays_up.dtype)
                                         for i in range(self.number_of_TLS)]

        for i in range(self.number_of_TLS):
            if self.IsInUpperState[i]:
                self.TLS_delays[i][0::2] = self.TLS_delays_down[i]
                self.TLS_delays[i][1::2] = self.TLS_delays_up[i]
            else:
                self.TLS_delays[i][0::2] = self.TLS_delays_up[i]
                self.TLS_delays[i][1::2] = self.TLS_delays_down[i]


    def generate_temporal_track_of_transition_frequency(self):
    # Create sequence function v(t) where t is time and v is the frequency
    # of electronic transition.
    # Don't forget to note the sequence of TLS hops as they happen

        # set initial states for slow TLSs
        TLS_state = self.IsInUpperState[self.slow_TLSes]

        #fill the list of next jump times based on initial state vector
        TLS_next_jump_time = np.zeros(self.number_of_slow_TLS,
                                      dtype = np.float64)

        # Create initial jump times
        for i, k in enumerate(self.slow_TLSes):
            if self.IsInUpperState[k]:
                K = self.VLE.TLS_Kdown[k]
                TLS_next_jump_time[i] = np.random.exponential(scale = 1/K)
            else:
                K = self.VLE.TLS_Kup[k]
                TLS_next_jump_time[i] = np.random.exponential(scale = 1/K)

        # This is local variable that represents real time in experiment
        t = np.float64(0)

        # calculate slow TLS splittings
        nu = np.zeros(self.number_of_slow_TLS, dtype = np.float64)
        for i, k in enumerate(self.slow_TLSes):
            nu[i] = self.VLE.alpha*\
                        self.VLE.TLS_A[k]/self.VLE.TLS_E[k]*\
                        self.VLE.TLS_epsilon[k]/(self.VLE.TLS_r[k])**3

        #Find first point of v(t)
        v = {0.000: sum(nu*TLS_state)}
        nu_t = [[0.00,0.00]]
        # For every TLS do:
        # if next time in TLS delays sequence plus this TLS's last jump moment
        # minus last whole-sequence jump moment is the minimal such value across
        # all TLSes (i.e. this TLS is the first one to flip since last flip
        # observed anywhere in the entire ensemble of TLSes)
        # then mark this TLS as flipped, record this new moment as
        # its last flip time, add a new point to the sequence v(t) and
        # record ID of this TLS into the log of TLS jumps.

        # REPEAT UNTIL THE TIME IS OVER
        local_counter = 0
        while t < self.max_time:
            #find out what TLS flips earlier than all others
            index_of_TLS_that_flipped = np.argmin(TLS_next_jump_time)
            t = TLS_next_jump_time[index_of_TLS_that_flipped]

            # Invert the state of this TLS
            TLS_state[index_of_TLS_that_flipped] = \
                                    not TLS_state[index_of_TLS_that_flipped]

            # Random-generate the new next jump time for this TLS
            if TLS_state[index_of_TLS_that_flipped]:
                K = self.VLE.TLS_Kdown[self.slow_TLSes[index_of_TLS_that_flipped]]
            else:
                K = self.VLE.TLS_Kup[self.slow_TLSes[index_of_TLS_that_flipped]]

            TLS_next_jump_time[index_of_TLS_that_flipped] += \
                        np.random.exponential(scale = 1/K)

            # Calculate current transition frequency of chromophore
            v_now = sum(nu*TLS_state)
            # Write it into memory data structures
            v[t] = v_now
            nu_t.append([t, v_now])
            
            local_counter += 1

            if local_counter%100000 == 0:
                print('{0}: current time point = {1:.2f}'.format(
                    self.instance_id,
                    t))

        #self.v = v
        self.nu_t = np.array(nu_t, dtype = np.float64)

        # After this is dome, v(t) will occupy a major amount of memory.
        # Save it to disk, along with entire jump log, params, what have you.


    def visualize_temporal_track_of_transition_frequency(self):
        x = sorted(list(self.v.keys()))
        y = [self.v[z] for z in x]

        #someplot = plt.plot(x,y, linestyle='-', drawstyle='steps-pre')
        #plt.ioff()
        fig = plt.figure(1)
        plt.axis((-15, 15, self.max_time, 0))
        plt.plot(y,x,'_', markersize=3, alpha=1, color = 'black')
        plt.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
            
        #plt.axis('off')
        plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        #plt.close(fig)

    def plot_scans(self, verboseMode = False, path_to_file = 'scans.png'):
        """Procedure that plots scans as they look in real experiment"""
#        # Then, use this v(t) to calculate the scans of virtual experimental setup
#        # as follows:
#
#        # For each time interval within microstep, where v(t) is being constant,
#        # sample the Initial_fast-broadened_spectrum by v(t) and sample it at the
#        # current laser frequency. Multiply the sampled intensity by the length of
#        # time interval where v(t) takes this value and doesn't change.
#        # Sum all multiplication results, divide by microstep length
#        # and it would be the average intensity
#        # gathered at this microstep at the laser wavelength where the laser has
#        # been during this microstep.
#        # Do this for all microsteps until the ending time of experiments.

        #v_time_points = sorted(self.v.keys())

        number_of_scans =\
                    math.floor(self.max_time/(self.steps_in_scan*self.secs_per_step))
        self.number_of_scans = number_of_scans
        
        point_before_this_step = 0
        nu_before_this_step = 0
        
        scans = np.zeros([number_of_scans, self.steps_in_scan])
        
        # make a precalculated list of lists that are internal points of
        # each step
        internal_points_for_all_steps = \
                    tuple([[] for i in range(number_of_scans*self.steps_in_scan)])
        for r in self.nu_t:
            try:
                internal_points_for_all_steps[int(r[0]//self.secs_per_step)]\
                                                                .append(r)
            except IndexError:
                #Means that we came up to a time that is beyond the last scan
                break
        
        # iterate over scans and their steps
        for scan_index in range(number_of_scans):
            if verboseMode:
                print("Scan {0}".format(scan_index))
                    
            for step_index in range(self.steps_in_scan):                
                step_begin = (scan_index*self.steps_in_scan + step_index)/ \
                                                            self.steps_per_sec
                step_end = (scan_index*self.steps_in_scan + step_index + 1)/ \
                                                            self.steps_per_sec

                this_point_wavelength = self.spectral_range/self.steps_in_scan* \
                                                                    step_index
                 
                #Before optimization this line was:
                #   internal_points = [t for t in v_time_points \
                #                    if ((t < step_end) and (t >= step_begin))]
                # 
                # then it was optimized and now the list of internal_points
                # for each step is precalculated
                internal_points = internal_points_for_all_steps[step_index + 
                                                scan_index*self.steps_in_scan]

                if not internal_points:
                    this_point_signal = \
                        self.broadened_spectrum(this_point_wavelength - \
                            nu_before_this_step)*self.secs_per_step
                            
                    scans[scan_index, step_index] = \
                            this_point_signal/self.secs_per_step      
                    continue
                    
                #First point
                this_point_signal = \
                 self.broadened_spectrum(this_point_wavelength - \
                 nu_before_this_step)* \
                 (internal_points[0][0] - step_begin)
                
                #Middle point but the first and the last
                for i, internal_point in enumerate(internal_points[0:-1]):
                    this_point_signal += \
                     self.broadened_spectrum(this_point_wavelength - \
                     internal_points[i][1])* \
                     (internal_points[i+1][0] - internal_points[i][0])
                    
                #Last point
                this_point_signal += \
                 self.broadened_spectrum(this_point_wavelength - \
                 internal_points[-1][1])* \
                 (step_end - internal_points[-1][0])
                
                scans[scan_index, step_index] = \
                            this_point_signal/self.secs_per_step                
                
                nu_before_this_step = internal_points[-1][1]
        
        self.Scans = scans
        
        # Show and save to disk
        np.savetxt('{0}.txt'.format(self.instance_id), scans)
        plt.ioff()
        fig = plt.figure(self.instance_id, figsize=(6,10))        
        plt.xlabel('Laser detuning, GHz', fontsize=22)
        plt.ylabel('Time, min', fontsize=22)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.imshow(scans, aspect='auto', 
                   cmap = plt.cm.Greys, 
                   interpolation='nearest',
                   # vmin=0, vmax=3e5,
                   extent=[0,30, self.max_time/60,0])
        fig.savefig(path_to_file, dpi=100, bbox_inches = 'tight')

    def simulate_one_run(self, verbose = False, path_to_file = 'scans.png'):
        self.subdivide_TLSes_by_speeds()
        if verbose:
            print("SubdivideBySpeeds - DONE")

        self.calculate_initial_fast_broadened_spectrum(plotting = False, 
                                                  verbose = verbose)
        if verbose:
            print("Calculate Broadened spectrum - DONE")
        self.generate_initial_states_for_TLSs()
        self.generate_temporal_track_of_transition_frequency()
        if verbose:
            print("generate_temporal_track_of_transition_frequency - DONE")
        self.plot_scans(verboseMode = False, path_to_file = path_to_file)
        
    def simulate_many_runs(self, N = 10, RefreshVLE = True, verbose = False):
        for i in range(N):
            print('Simulation run № {0}'.format(i))
            
            if RefreshVLE:
                self.VLE = VirtualLocalEnvironment()
                if verbose:
                    print('VLE is refreshed.')
                self.subdivide_TLSes_by_speeds()
                if verbose:
                    print("SubdivideBySpeeds - DONE")
        
                self.calculate_initial_fast_broadened_spectrum(plotting = False, 
                                                           verbose = verbose)
                if verbose:
                    print("Calculate Broadened spectrum - DONE")
                    
            self.generate_initial_states_for_TLSs()
            self.generate_temporal_track_of_transition_frequency()
            if verbose:
                print("generate_temporal_track_of_transition_frequency - DONE")
            self.plot_scans(verboseMode = False, 
                           path_to_file = '{0:02d}.png'.format(i))

    def get_one_width(self, 
                    experiment_time = 0):
        if not experiment_time:
            experiment_time = 1/self.threshold_between_fast_and_slow
        else:
            self.threshold_between_fast_and_slow = 1/experiment_time
        self.subdivide_TLSes_by_speeds()
        width = self.calculate_initial_fast_broadened_spectrum(plotting = False, 
                                                               verbose = False)
        #print("Calculate Broadened spectrum - DONE")
        return width     

    def get_widths_distribution(self, 
                              experiment_time = 0,
                              sample_size = 600,
                              verbose = False,
                              do_cutting = False,
                              distance_to_surface = 10,
                              surface_TLS_density = 2,
                              filename = 'widths'):
        import gc
        import json

        if not experiment_time:
            experiment_time = 1/self.threshold_between_fast_and_slow
        else:
            self.threshold_between_fast_and_slow = 1/experiment_time
        
        start_time = time.time()
        widths = []
        for i in range(sample_size):
            if i > 0:
                time_remaining = \
                    (time.time() - start_time)/i*(sample_size - i)/60
            else:
                time_remaining = 0
                
            print('                                      >>>>>>> '
                  '{0}% completed. {1:.1f} minutes left.'.format(
                int(100*i/sample_size),
                time_remaining))
            
            # Generate new VLE
            # Randomize the distance to surface according to 
            # Fick's diffusion solution for diffusion length L
            L = distance_to_surface
            depth_of_molecule = np.abs(np.random.normal(0, 
                                                        L/np.sqrt(2), 
                                                        size=1))[0]
            print('Current depth of molecule = {0:.2f} nm'.format(
                depth_of_molecule))
            

            
            self.VLE = VirtualLocalEnvironment(
                do_cutting = do_cutting,
                distance_to_surface = depth_of_molecule,
                surface_TLS_density = surface_TLS_density)
                
            self.number_of_TLS = self.VLE.number_of_TLS
            if verbose:
                print('VLE is refreshed.')
                
            width = self.get_one_width(experiment_time) 
            gc.collect()
            
            # if width is zero then something went wrong. Ignore it.
            if width:
                widths.append(width)
                print('Widths is {0:.2f}. Saved.'.format(width))
            
                with open(filename+'.json', 'w') as outfile:
                    json.dump(widths, outfile)
                
        #plot hist
        plt.ioff()
        plt.hist(widths, bins = OptimalBins(widths))
        plt.savefig(filename + '_histogram.png', 
                    dpi=100, 
                    bbox_inches = 'tight')


#==============================================================================
# MAIN PROGRAM BODY
#==============================================================================


#def main():    
def GetWidths(r, density, filename):
    sim = MoleculeSpectrumSimulator(do_cutting = True,
                                    distance_to_surface = r)
    sim.get_widths_distribution(experiment_time = 77.5, 
                                do_cutting = True,
                                distance_to_surface = r,
                                surface_TLS_density = density,
                                filename = filename)
                                
#for density in [1]:
#    for depth in [13.4, 7, 3.6]:
#        print('DENSITY = {0}, DEPTH = {1}'.format(density, depth))
#        GetWidths(depth, density, 
#                  'density_{0}__depth_{1}'.format(density, depth)) 
                                

def GetScans(instance_id):
    sim = MoleculeSpectrumSimulator(do_cutting = False)
    sim.instance_id = instance_id
    print("Initiated - DONE")
    sim.subdivide_TLSes_by_speeds()
    print("SubdivideBySpeeds - DONE")
    width = sim.calculate_initial_fast_broadened_spectrum(plotting = False, 
                                                          verbose = True)
    print(width)
    sim.generate_initial_states_for_TLSs()
    sim.generate_temporal_track_of_transition_frequency()
    sim.plot_scans(path_to_file = 'scans_{0}.png'.format(instance_id))
    
    with open('sim_{0}.pickle'.format(instance_id), 'wb') as outfile:
        pickle.dump(sim, outfile)
        outfile.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Instance_number.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                       help='an integer for instance number.')
    
    args = parser.parse_args()
    instance_id = vars(args)['integers'][0] 
    
    GetScans(instance_id)

#from multiprocessing import Pool
#
#if __name__ == '__main__':
#    with Pool(8) as p:
#        print(p.map(GetScans, [1, 2, 3, 4, 5, 6, 7, 8]))
                                



    #
    #pr.enable()

#    print("Calculate Broadened spectrum - DONE") 

#if __name__ == '__main__':
#    main()