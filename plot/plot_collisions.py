from amuse.lab import *
import numpy as np
from matplotlib import pyplot as plt

def plot_cumulative_collision_rate(dir):
    filename = dir + "/out.data"
    f = open(filename)
    r = f.readlines()
    nsnapshot = 0
    tcoll = [] | units.Myr
    for ri in r:
        if "At time" in ri:
            st = ri.split()
            print(st)
            tcoll.append(float(st[2]) | units.Myr)
    f.close()
    #tcoll.append(0.0|units.Myr)
    tcoll.append(1.0|units.Myr)

    t = np.sort(tcoll.value_in(units.Myr))
    rate = len(t) # number of collisions per Myr
    print(len(t))
    f = np.linspace(0, rate, rate)
    print(t, f)
    return t, f


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-f", 
                      dest="filename", default = "out.data",
                      help="input filename [%default]")
    return result
    
if __name__=="__main__":
    o, arguments  = new_option_parser().parse_args()


    dirs_ISF = ["run_ISF/FrN2500n300isf_a25+1000R0.25pcQ05_R1",
            "run_ISF/FrN2500n300isf_a25+1000R0.5pcQ05_R1",
            "run_ISF/FrN2500n300isf_a25+1000R1.0pcQ05_R1"]
    #dirs_FFC = ["run_FFC/FrN2500n600x1.2ffc_R0.25pcQ05_R1",
    #        "run_FFC/FrN2500n600x1.2ffc_R0.5pcQ05_R1",
    #        "run_FFC/FrN2500n600x1.2ffc_R1.0pcQ05_R1"]
    dirs_FFC = ["run_FFC/PlN2500n600x1.2m1ffc_R0.25pcQ05_R1",
            "run_FFC/PlN2500n600x1.2m1ffc_R0.5pcQ05_R1",
            "run_FFC/PlN2500n600x1.2m1ffc_R1.0pcQ05_R1"]
    dirs = dirs_ISF + dirs_FFC
    label = ["R=0.25pc", "R=0.5pc", "R=1.0pc"]
    fig = plt.figure(figsize=(6,5))    
    plt.rcParams.update({'font.size': 12})
    c = ['b', 'orange', 'g', 'b',
         'orange', 'g']
    for i in range(len(dirs)):
        t, f = plot_cumulative_collision_rate(dirs[i])
        if "ISF" in dirs[i]:
            plt.plot(t, f, label = label[i], lw=4, c=c[i])
        else:
            plt.plot(t, f, lw=2, ls="-.", c=c[i])

    plt.legend()
    plt.semilogx()
    plt.xlabel("$t [Myr]$")
    plt.ylabel("$f_{<t}$")
    plt.savefig("fig_collision_evolution_ISF_Fr.pdf")
    plt.show()


