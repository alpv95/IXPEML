"""Post processing on tracks: Makes fig.4 from paper -- mu100, mu0 and phi0 for each energy bin"""
#%%
home_dir = '/home/groups/rwr/alpv95/tracksml/'
import os, sys
sys.path.append(home_dir)
from util.net_test import *
import argparse
from util.methods import *
from itertools import tee
from matplotlib import rcParams
rcParams['savefig.dpi']           = 300
rcParams['path.simplify']         = True
rcParams['figure.figsize']        = 4,4
rcParams['font.family']           = "serif"
rcParams['mathtext.fontset']      = "custom"
rcParams['errorbar.capsize']      = 3
rcParams['axes.linewidth']        = 2 
rcParams['font.weight']           = "bold"
rcParams['xtick.major.size']      = 6
rcParams['ytick.major.size']      = 6 
rcParams['xtick.minor.size']      = 3   
rcParams['ytick.minor.size']      = 3
rcParams['xtick.direction']      = "in"
rcParams['ytick.direction']      = "in" 
rcParams['xtick.top']      = True
rcParams['ytick.right']      = True 
rcParams['xtick.major.width']     = 1
rcParams['ytick.major.width']     = 1
rcParams['xtick.minor.width']     = 1
rcParams['ytick.minor.width']     = 1
rcParams['lines.markeredgewidth'] = 1 
rcParams['legend.numpoints']      = 1
rcParams['legend.frameon']        = False
rcParams['legend.handletextpad']      = 0.3
import matplotlib.pyplot as plt
#%%
parser = argparse.ArgumentParser()
parser.add_argument('file_pol', type=str,
                    help='Pickle file to postprocess')
parser.add_argument('file_unpol', type=str,
                    help='Pickle file to postprocess')
args = parser.parse_args()

#%%
def main():
    with open(home_dir + args.file_pol, "rb") as file:
        A = pickle.load(file)
    angles_pol, angles_mom_pol, angles_sim_pol, moms_pol, errors_pol, _, _, _, \
    _, energies_sim_pol, _, _, _ = A

    with open(home_dir + args.file_unpol, "rb") as file:
        B = pickle.load(file)
    angles, angles_mom, angles_sim, moms, errors, _, _, _, \
    _, energies_sim, _, _, _ = B

    E = np.sort(list(set(energies_sim)))
    assert (E == np.sort(list(set(energies_sim_pol)))).all()

    mus_pol, phis = pol_E(E, energies_sim_pol, angles_pol, angles_mom_pol, angles_sim_pol, moms_pol, errors_pol)
    mus_unpol= unpol_E(E, energies_sim, angles, angles_mom, angles_sim, moms, errors)

    np.save("paper_plot4_3",(mus_pol, phis, mus_unpol))

    plot(np.stack([m.T * 100 for m in mus_pol],axis=0), np.stack([m.T * 100 for m in mus_unpol],axis=0), np.stack([m.T for m in phis],axis=0))

def unpol_E(E, energies_sim, angles, angles_mom, angles_sim, moms, errors,):
    t = NetTest(n_nets=14)
    mus = []
    for e1, e2 in [(E[1],E[6]),(E[6],E[11]),(E[11],E[16]),(E[16],E[21]),(E[21],E[26]),
              (E[26],E[31]),(E[31],E[36]),(E[36],E[41]), (E[41],E[46]),
            (E[46],E[51]),(E[51],E[56]),(E[56],E[61]),(E[61],E[66]),(E[66],E[-1])]:

        cut = (e1 <= energies_sim) * (e2 >= energies_sim)
        A1 = (np.ndarray.flatten(angles[cut]), angles_mom[cut], angles_sim[cut], moms[cut], np.ndarray.flatten(errors[cut]), [None], [None], 
        [None], [None], [None], [None], [None],[None])
        mu, _, mu_err, _ = t.fit_mod(A1, method='stokes')
        mu1, _, mu_err1, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1)
        mu2, _, mu_err2, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=2)
        mus.append(np.stack([np.concatenate([mu,mu_err]),np.concatenate([mu1,mu_err1]),np.concatenate([mu2,mu_err2])],axis=1))
        print(e1,e2,mu,mu1,mu2)

    return mus

def pol_E(E, energies_sim, angles, angles_mom, angles_sim, moms, errors,):
    t = NetTest(n_nets=14)
    ee = pairwise(E)
    mus = []
    phis = []
    chunks = []
    for e1, e2 in ee:
        print(e1,e2,)
        cut = (e1 <= energies_sim) * (e2 >= energies_sim)
        A1 = (np.ndarray.flatten(angles[cut]), angles_mom[cut], angles_sim[cut], moms[cut], np.ndarray.flatten(errors[cut]), [None], [None], 
          [None], [None], [None], [None], [None],[None])
        chunks.append(A1)
        mu, phi0, mu_err, phi0_err = t.fit_mod(A1, method='stokes')
        mu1, phi01, mu_err1, phi0_err1 = t.fit_mod(A1, method='weighted_MLE', error_weight=1)
        mu2, phi02, mu_err2, phi0_err2 = t.fit_mod(A1, method='weighted_MLE', error_weight=2)
        print(e1,e2,mu,mu1,mu2)
        mus.append(np.stack([np.concatenate([mu,mu_err]),np.concatenate([mu1,mu_err1]),np.concatenate([mu2,mu_err2])],axis=1))
        phis.append(np.stack([np.concatenate([phi0,phi0_err]),np.concatenate([phi01,phi0_err1]),np.concatenate([phi02,phi0_err2])],axis=1))

    return mus, phis

def plot(mus_pol, mus_unpol, phis):
    X1 = np.arange(1.05,8.0,0.1)
    X = np.arange(1.35,8.1,0.5)

    Y1 = [mus_pol[:,0,0],mus_pol[:,0,1],mus_pol[:,0,2],mus_pol[:,2,0],mus_pol[:,1,1],mus_pol[:,1,0]]
    Yerr1 = [mus_pol[:,0,0+3],mus_pol[:,0,1+3],mus_pol[:,0,2+3],mus_pol[:,2,0+3],mus_pol[:,1,1+3],mus_pol[:,1,0+3]]

    Y = [mus_unpol[:,0,0],mus_unpol[:,0,1],mus_unpol[:,0,2],mus_unpol[:,2,0],mus_unpol[:,1,1],mus_unpol[:,1,0]]
    Yerr = [mus_unpol[:,0,0+3],mus_unpol[:,0,1+3],mus_unpol[:,0,2+3],mus_unpol[:,2,0+3],mus_unpol[:,1,1+3],mus_unpol[:,1,0+3]]

    fig, ax = plt.subplots(nrows=3,sharex=True,gridspec_kw={"hspace":0.0},figsize=(7,10))

    errbr0 = ax[0].errorbar(X1,Y1[2] ,color='k',yerr=Yerr1[2], label= r"$\mu$ at $\Pi = 1$",linestyle='dotted')
    errbr1 = ax[0].errorbar(X1, Y1[3], yerr=Yerr1[3],color='r',linestyle='--', label=r"NN w/ weights 2",marker="o",markersize=3.5)
    errbr2 = ax[0].errorbar(X1, Y1[5], yerr=Yerr1[5],color='cyan',linestyle='--', label=r"NN w/ weights 1",marker="o",markersize=3.5)
    errbr4 = ax[0].errorbar(X1,Y1[0] ,color='r',yerr=Yerr1[0], label= r"NN",linestyle='solid')
    errbr5 = ax[0].errorbar(X1,Y1[4], yerr=Yerr1[4],color='b',linestyle='--', label=r"Mom. w/ cuts",fmt="o",markersize=3.5)
    errbr6 = ax[0].errorbar(X1,Y1[1] , color='b',yerr=Yerr1[1], label= r"Mom.",linestyle='solid')

    ax[0].errorbar(2.7, 30.5, yerr=0.2, marker="x", markersize="6",color="k")
    ax[0].errorbar(4.5, 53.4, yerr=0.2, marker="x", markersize="6",color="k")
    ax[0].errorbar(6.4, 68.6, yerr=0.2, marker="x", markersize="6",color="k")
    ax[0].errorbar(8.0, 73.8, yerr=0.2, marker="x", markersize="6",color="k")

    leg = ax[0].legend(loc = 2,)
    lines = [errbr0.lines[0],errbr1.lines[0],errbr2.lines[0],errbr4.lines[0],errbr5.lines[0],errbr6.lines[0]]
    for line, text in zip(lines, leg.get_texts()):
        text.set_color(line.get_color())
    ax[0].set_ylabel(r"Modulation [%]",fontweight="bold")
    ax[0].set_ylim(0.1,100.0)
    ax[0].set_xlim(1.0,7.6)
    ax[0].minorticks_on()

    errbr0 = ax[1].errorbar(X,Y[2] ,color='k',yerr=Yerr[2], label= r"$\mu$ at $\Pi = 0$",linestyle='dotted')
    errbr1 = ax[1].errorbar(X, Y[5], yerr=Yerr[5],color='cyan',linestyle='--', label=r"NN w/ weights 1",marker="o",markersize=3.5)
    #errbr2 = ax[1].errorbar(X[0], Y[6], yerr=Yerr_bootstrap[0],color='m',linestyle='--', label=r"NN w/ weights 3",marker="o",markersize=3.5)
    errbr3 = ax[1].errorbar(X, Y[3], yerr=Yerr[3],color='r',linestyle='--', label=r"NN w/ weights 2",marker="o",markersize=3.5)
    errbr4 = ax[1].errorbar(X,Y[0] ,color='r',yerr=Yerr[0], label= r"NN",linestyle='solid')
    errbr5 = ax[1].errorbar(X,Y[4], yerr=Yerr[4],color='b',linestyle='--', label=r"Mom. w/ cuts",fmt="o",markersize=3.5)
    errbr6 = ax[1].errorbar(X,Y[1] , color='b',yerr=Yerr[1], label= r"Moments",linestyle='solid')

    ax[1].errorbar(2.7, 0.8, yerr=0.2, marker="x", markersize="6",color="k")
    ax[1].errorbar(4.5, 0.8, yerr=0.2, marker="x", markersize="6",color="k")
    ax[1].errorbar(6.4, 1.1, yerr=0.2, marker="x", markersize="6",color="k")
    ax[1].errorbar(8.0, 0.6, yerr=0.2, marker="x", markersize="6",color="k")

    #ax[1].set_xlabel(r"Energy [keV]",fontweight="bold")
    ax[1].set_ylabel(r"Modulation [%]",fontweight="bold")
    ax[1].set_ylim(0.0,1.4)
    ax[1].set_xlim(1.0,7.6)
    ax[1].minorticks_on()

    # name = "phi"
    # X,Y,Yerr,Yerr_bootstrap = np.load("plots/" + name + ".npy",allow_pickle=True)
    Y = [phis[:,0,0],phis[:,0,1],phis[:,0,2],phis[:,2,0],phis[:,1,1],phis[:,1,0]]
    Y = [np.sqrt(pi2_pi2(y - np.pi/2)**2) * 180/np.pi for y in Y]
    Yerr = [phis[:,0,0+3],phis[:,0,1+3],phis[:,0,2+3],phis[:,2,0+3],phis[:,1,1+3],phis[:,1,0+3]]

    errbr0 = ax[2].errorbar(X1,Y[2]  ,color='k',yerr=Yerr[2], label= r"$\mu$ at $\Pi = 1$",linestyle='dotted')
    errbr1 = ax[2].errorbar(X1, Y[3], yerr=Yerr[3] ,color='r',linestyle='--', label=r"NN w/ weights", marker='o',markersize=3.5)
    errbr2 = ax[2].errorbar(X1,Y[0] ,color='r',yerr=Yerr[0], label= r"NN",linestyle='solid')
    errbr3 = ax[2].errorbar(X1,Y[4], yerr=Yerr[4],color='b',linestyle='--', label=r"Mom. w/ cuts",marker='o',markersize=3.5)
    errbr4 = ax[2].errorbar(X1,Y[1] , color='b',yerr=Yerr[1], label= r"Moments",linestyle='solid')
    ax[2].set_xlabel(r"Energy [keV]",fontweight="bold")
    ax[2].set_ylabel(r"$\rm{RMSE_{\phi}}$ [deg]",fontweight="bold")
    # ax.set_ylim(0.0,6)
    ax[2].set_xlim(1.0,7.6)
    ax[2].set_ylim(-0.4,2.45)
    ax[2].minorticks_on()

    plt.savefig("plots/" + "modulation_joint4" + ".pdf",format="pdf")

if __name__ == '__main__':
    main()
