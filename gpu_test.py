'''
Wrapper file to apply an ensemble of NNs to square track datasets.
'''

from util.net_test import *
import os, sys
import argparse
from util.definitions import ensembles

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble', type=str,choices=["flat_weight","bessel_rand_small"],
                    help='Which network ensemble to use: Ensemble prediction or single prediction')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot histograms and modulation curves')
parser.add_argument('--bayes', action='store_true',
                    help='Whether to sample from dropout')
parser.add_argument('--cut', type=float, default=0.83,
                    help='What proportion of tracks to leave after ellipticity cut')
parser.add_argument('--save', type=str, default=None,
                    help='Save Net FoV results in Pandas dataframe with this name')
parser.add_argument('--method', type=str, default="stokes",
                    help='Which modulation factor/angle fitting method to use on final set of predicted track angles')
parser.add_argument('--model_list', type=str, nargs='+', default=[],
                    help='List of Models to evaluate')
parser.add_argument('--data_list', type=str, nargs='+',
                    help='List of Data to evaluate on')
parser.add_argument('--datatype', type=str, default="sim", choices=["sim","meas"],
                    help='Simulated or Measured track data')
parser.add_argument('--input_channels', type=int, default=2,
                    help='Number of input channels to network')
parser.add_argument('--stokes_correct', type=float, choices=[2.0, 2.3, 2.7, 3.1, 3.7, 5.9], default=None,
                    help='Whether to correct measured tracks for spurious modulation')
args = parser.parse_args()


def main(**kwargs):
    ensemble=kwargs["ensemble"]; plot=kwargs["plot"]; bayes=kwargs["bayes"]; fitmethod=kwargs["fitmethod"]; cut=kwargs["cut"]
    save_table = kwargs["save_table"]; model_list = [M + '/models' for M in kwargs["model_list"]]; data_list = kwargs["data_list"]
    input_channels = kwargs["input_channels"]; stokes_correct = kwargs["stokes_correct"]; datatype = kwargs["datatype"]


    if ensemble:
        net_list = ensembles[ensemble]
        print("Evaluating using ensemble: \n", net_list)
        print("\n {} NNs in the ensemble \n".format(len(net_list)))
        t = NetTest(nets=net_list, fitmethod=fitmethod, datasets=data_list, plot=plot, n_nets=len(net_list), cut=cut, datatype=datatype,
                    save_table=save_table, input_channels=input_channels, stokes_correct=stokes_correct)
        t.ensemble_predict(bayes=bayes)

    else:
        print("Evaluating models: \n", model_list)
        net_list = []

        extensionsToCheck = ["191","171","151",'131','111','91']
        for model in model_list:
            nets = os.listdir(os.path.abspath(os.path.join("data/nn/", model)))
            for net in nets:
                if any(ext in net for ext in extensionsToCheck):
                    net_list.append(os.path.join(model, net))

        t = NetTest(nets=net_list, fitmethod=fitmethod, datasets=data_list, plot=plot, cut=cut, 
                    save_table=save_table, input_channels=input_channels)
        t.single_predict(bayes=bayes)


if __name__ == '__main__':
    main(ensemble=args.ensemble, fitmethod=args.method, plot=args.plot, bayes=args.bayes, cut=args.cut, datatype=args.datatype,
        save_table=args.save, model_list=args.model_list, input_channels=args.input_channels,
        data_list=args.data_list, stokes_correct=args.stokes_correct)



