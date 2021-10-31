import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from ml_siamese import OneshotLoader
from gen_datasets import DatasetLoader
from constants import input_image_types

run = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
dataset_path = "./tmp"
run_path = os.path.join(dataset_path, run)

ways = np.arange(1, 9, 1)
trials = 100

def nearest_neighbour_correct(pairs, targets):
    """
    Returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)
    """
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways, n_trials, loader):
    """
    Returns accuracy of one shot
    """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))

    n_right = 0
    
    for i in range(n_trials):
        pairs,targets = loader.make_task(N_ways,"val")
        correct = nearest_neighbour_correct(pairs,targets)
        n_right += correct
    return 100.0 * n_right / n_trials

def compare_all(save_path, accs): 
    fig, ax = plt.subplots()
    
    for name in accs: 
        plt.plot(ways, accs[name], "", label=name)
    
    ax.legend()
    
    results_path = os.path.join(save_path, "all")
    plt.savefig(results_path)
        
def compare(loader, model): 
    fig, ax = plt.subplots()
    
    val_accs, train_accs, nn_accs = [], [], []

    for N in ways:
        val_accs.append(loader.test(model, N, trials, "val", verbose=True))
        train_accs.append(loader.test(model, N, trials, "train", verbose=True))
        nn_accs.append(test_nn_accuracy(N, trials, loader))

    
    
    plt.plot(ways, val_accs, "m", label="validation")
    plt.plot(ways, train_accs, "y", label="training")
    plt.plot(ways, nn_accs, "c", label="nearest neighbor")
    plt.plot(ways, 100.0 / ways, "r", label="random")
    
    ax.legend()
    
    results_path = os.path.join(loader.results_folder, loader.spectrogram_type)
    plt.savefig(results_path, dpi=100)
    
    return val_accs
        
def main(): 
    skip = [""]
    accs = {}
    results_folder = ""
    
    for spectrogram_type in input_image_types: 
        if spectrogram_type in skip:
            print("skipping: ", spectrogram_type) 
            continue
        
        print("running ML analysis on spectrogram type: ", spectrogram_type)
        
        dl = DatasetLoader(spectrogram_type, run)
        dl.generate()
        
        data_path = os.path.join(run_path, "images", spectrogram_type) + "/"

        # Needed to fix some tensorflow compilation errors
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        loader = OneshotLoader(data_path, spectrogram_type, run_path)
        if results_folder == "": 
            results_folder = loader.results_folder
        
        model = loader.train()
        
        # save validation accuracies
        accs[spectrogram_type] = compare(loader, model)
        break
    
    compare_all(results_folder, accs)
        
main()
