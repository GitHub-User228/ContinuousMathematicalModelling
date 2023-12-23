import matplotlib.pyplot as plt
import seaborn as sns



def simple_plot(x, y, xlabel=None, ylabel=None, scatterplot=False, lineplot=True,
                params_dict=None, xticks=None, yticks=None, rotation=0):
    if params_dict is not None:
        plt.rcParams.update(params_dict)
    plt.figure()
    sns.set_style("darkgrid")
    if lineplot: sns.lineplot(x=x, y=y)
    if scatterplot: sns.scatterplot(x=x, y=y)
    if xlabel != None: plt.xlabel(xlabel)
    if ylabel != None: plt.ylabel(ylabel)
    if xticks != None: plt.xticks(xticks, rotation=rotation)
    if yticks != None: plt.yticks(yticks) 
    if rotation != None: plt.xticks(rotation=rotation)
    plt.show()  



def multiplot(X, Y, xlabel=None, ylabel=None, labels=None, scatterplot=False, lineplot=True,
              params_dict=None, xticks=None, yticks=None, rotation=0, yscale='linear'):
    if params_dict is not None:
        plt.rcParams.update(params_dict)
    plt.figure()
    sns.set_style("darkgrid")
    if lineplot: 
        for (x, y, label) in zip(X, Y, labels):
            sns.lineplot(x=x, y=y, label=label)
    if scatterplot: 
        for (x, y, label) in zip(X, Y, labels):
            sns.scatterplot(x=x, y=y)
    if xlabel != None: plt.xlabel(xlabel)
    if ylabel != None: plt.ylabel(ylabel)
    if xticks != None: plt.xticks(xticks, rotation=rotation)
    if yticks != None: plt.yticks(yticks) 
    if rotation != None: plt.xticks(rotation=rotation)
    plt.yscale(yscale)
    plt.show()  