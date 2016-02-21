# -*- coding: utf-8 -*-
"""
Plotter of subsurface linewidths and Mighty Comparer to
simulated results.

Created on Mon Jun 22 14:40:19 2015

@author: god
"""
import numpy as np
import matplotlib.pyplot as plt
import json

simfolder = 'width_distributions_at_complex_VLEs/exponent_surface_layers/all/'
binwidth = 0.1

depths = [0.6, 1.6, 3.6, 7.0, 13.4, 29.5]
depths1 = [0.6, 1.6, 3.6, 7.0, 14.3, 29.5]

exp_hists = [np.loadtxt('exp_depth_{0:.1f}nm.txt'.format(d), 
                delimiter = '\t') \
             for d in depths]
                 
#get sims
def plot_with_sim(expthickness, density):
    sim_widths = []
    for d in depths1:
        try:
            with open(simfolder + \
                      'expthick{0}_density_{1}__depth_{2:.1f}.json'. \
                      format(expthickness, density, d)) as in_file:
                sim_widths.append(json.load(in_file))
        except FileNotFoundError:
            print('FAILED expthick{0}_density_{1}__depth_{2:.1f}'. \
                      format(expthickness, density, d))
            sim_widths.append([])
             
    f, axarr = plt.subplots(len(depths), sharex=True)
    for i, hist in enumerate(exp_hists):                     
        if sim_widths[i]:
            axarr[i].hist(sim_widths[i], bins=np.linspace(0,20,20/binwidth+1),
                          alpha = 0.5, color = 'yellow',
                          histtype='stepfilled')
            #axarr[i].set_title('Depth={0}'.format(depths[i]), ha='right')
            axarr[i].bar(hist[:,0]/16.666, 
                         hist[:,1]/max(hist[:,1])*float(axarr[i].get_ylim()[1]), 
                         width = (hist[2,0] - hist[1,0])/16.666,
                         alpha = 0.3)
        else:
            axarr[i].bar(hist[:,0]/16.666, 
                         hist[:,1], 
                         width = (hist[2,0] - hist[1,0])/16.666,
                         alpha = 0.3)            
        axarr[i].text(0.99, 0.9, '{0:.1f} nm'.format(depths[i]),
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=axarr[i].transAxes,
                        size = 20)
        axarr[i].set_yticklabels([])
        axarr[i].set_ylim((0,1.3*axarr[i].get_ylim()[1]))
        
    f.savefig('exp{0:.1f}_d{1:.1f}.png'.format(expthickness, density), 
              transparent=True, 
              bbox_inches='tight', 
              pad_inches=0,
              dpi = 100)

densities = [3,4,5]
exps = [0.5,1,2,3,5,10]      
for d in densities:
    for e in exps:
        plot_with_sim(e, d)
        print('exp {0:.1f} density {1:.1f} - done.'.format(e,d))