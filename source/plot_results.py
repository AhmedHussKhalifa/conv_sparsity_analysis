import tensorflow as tf
import numpy as np
import sys
import argparse
from myconstants import *
import matplotlib.pyplot as plt
from sparsity_method_types import SparsityMethodTypes


# cmaps['Miscellaneous'] = [
#             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
#             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

# for More colors https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


def preprocessing(CR_txt, Density_txt, Modules_txt):    
    modules = []
    with open(Modules_txt, 'r') as input:
       for line in input:
            x = line.split()
            x = [int(i) for i in x] 
            modules.append(x)

    print("Modules Shape :", np.shape(modules))


    CPO_CR = CPS_CR = MEC_CR = CSCC_CR = SparesTensor_CR = np.empty(0,float)
    ru = ru_bound_mec = ru_bound_cscc                    = np.empty(0,float)
    
    with open(CR_txt, 'r') as input:
       for line in input:
            CPO_CR          = np.append(CPO_CR,             float(line.split()[conv_methods['CPO']-1]))
            CPS_CR          = np.append(CPS_CR,             float(line.split()[conv_methods['CPS']-1]))
            MEC_CR          = np.append(MEC_CR,             float(line.split()[conv_methods['MEC']-1]))
            CSCC_CR         = np.append(CSCC_CR,            float(line.split()[conv_methods['CSCC']-1]))
            SparesTensor_CR = np.append(SparesTensor_CR,    float(line.split()[conv_methods['SparseTensor']-1]))
    # CPO_CR = np.squeeze(CPO_CR)
    CR = Density = np.empty((0,np.shape(CPO_CR)[0]),float)
    
    CR = np.append(CR, [CPO_CR], axis=0)
    CR = np.append(CR, [CPS_CR], axis=0)
    CR = np.append(CR, [MEC_CR], axis=0)
    CR = np.append(CR, [CSCC_CR], axis=0)
    CR = np.append(CR, [SparesTensor_CR], axis=0)
    print("CR Shape :", CR.shape)


    with open(Density_txt, 'r') as input:
       for line in input:
            ru              = np.append(ru,                 float(line.split()[0]))
            ru_bound_mec    = np.append(ru_bound_mec,       float(line.split()[1]))
            ru_bound_cscc   = np.append(ru_bound_cscc,      float(line.split()[2]))
    Density = np.append(Density, [ru], axis=0)
    Density = np.append(Density, [ru_bound_mec], axis=0)
    Density = np.append(Density, [ru_bound_cscc], axis=0)
    print("Density Shape :", Density.shape)

    return modules, CR, Density

def get_cmap(n, name='rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_all_images(modules, CR, method, plots_dir):
    cmap = get_cmap(CR.shape[0])
    print(CR.shape)
    # set width of bar
    barWidth = 0.1
    img_num = CR.shape[1]/94
    conv_num = 94
    if (method is "CRs"):
        print("Running method is CR")
        md = 1
    elif (method is "Bounds"):
        print("Running method is BO")
        md = 10
    # for img_idx in range(0,CR.shape[1],conv_num):
    for img_idx in range(0,3*conv_num,conv_num):
        for mod in range(0,np.shape(modules)[0]):
            x = img_idx + np.array(modules[mod][:])
            # print(x)
            buff = CR[:, x ]
            # print(buff.shape)
            plt.figure()
            # Set position of bar on X axis
            r = []
            r.append(np.arange(buff.shape[1]))
            for i in range(1,buff.shape[0]):
                bar = [(x + barWidth) for x in r[i-1][:]]
                r.append(bar)
            plt.suptitle(method+" - mixed - %d"%abs(mod), fontsize=14)
            for i in range(0,buff.shape[0]):
                # print(md*(i+1))
                plt.bar(r[i][:], buff[i,:], facecolor=cmap(i), width=barWidth, edgecolor='white', label=SparsityMethodTypes.getModelByValue(md*(i+1)))

            plt.xlabel('Convolutions', fontweight='bold')
            plt.ylabel(method, fontweight='bold')
            convs = []
            for idx in range(0,len(modules[mod])):
                convs.append(("%d")%(idx+1))
            plt.xticks([r + barWidth for r in range(len(r[0][:]))], convs)
            plt.legend(prop={'size': 6})
            plt.savefig(plots_dir+method+' - ImgID %d - mixed_%d.png'%((img_idx/94)+1 ,mod), dpi= 600)
        # fig.close()
    return 1





def main():
    CR_txt        = FLAGS.gen_dir + "CR.txt" # compression Ratio (MEC/CSCC/CPO/CPS/SpareTensor)/Im2col
    Density_txt   = FLAGS.gen_dir + "density.txt"
    Modules_txt   = FLAGS.gen_dir + "Modules.txt"
    plots_dir     = FLAGS.plot_dir
    modules, CR, Density = preprocessing(CR_txt, Density_txt, Modules_txt)
    fig = plot_bars(modules, CR, "CRs", plots_dir)
    fig = plot_bars(modules, Density, "Bounds", plots_dir)
    # fig.show()

    # plot_bars(modules, Density)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--gen_dir',
      type=str,
      default='../gen/',
      help='generated texts directory'
  )

    parser.add_argument(
      '--plot_dir',
      type=str,
      default='../plots/',
      help='generated plots directory'
  )
    FLAGS, unparsed = parser.parse_known_args()
    main()