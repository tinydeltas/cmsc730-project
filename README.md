# cmsc730-project

## Getting started

**Requirements** 
Python 3.73+

May need twiddling to work on an M1 chip. 

1. Set up virtualenv using `setup.sh` or by running 

```
#!/bin/zsh

python3 -m venv venv 
source venv/bin/activate
pip install jupyter
ipython kernel install --name "local-venv" --user
python -m pip install -r requirements.txt
```

2. Run `00-gen_spectrograms.py`: Generates spectrogram images from the `.wav` raws. Stores them in `param_data_path` (see below). 

```
python 01-gen_spectrograms.py 
```

3. Run `01-pipeline.py`: Defines and trains input image types on 7-layer CNN Siamese network model. Takes about 10 minutes per image type, for total of ~1.5 hours to train and compare on every image type. 

```
python 02-pipeline.py 
```

## Directory overview 
- `data/`: Data for predefined gestures. 
    - `images`: The spectrogram images generated from the raw `.wav` files
    - `wav`: The raw `.wav` files of the eight predefined gestures, collected by the SEEED microphone mounted on a raspberry pi. 

- `pipeline/` 
    - `constants.py`: 

            -    `param_input_image_types`: Defines the input image (spectrogram) types we are interested in feeding to the ML model for training and validation. 
    
            - `param_default_gestures`: labels for the pre-defined gestures (corresponding to the names of their respective folders in `data/images`)

            - `param_loss_function`: Loss function for the ML model.
            **Default: `binary_crossentropy`** 
            
            - `param_optimizer`: Optimizer algorithm. 
            **Default: Adam (Stochastic gradient descent).**           
            
            - `param_N_way`: How many classes to assign a potential task to. 
            **Default: 8 (for the 8 pre-defined gestures)**          
            
            - `param_n_val`: How many tasks to validate on. 
            **Default: 7**           
            
            - `param_batch_size_per_trial`: Number of paired batch tasks per trial. 
            **Default: 7**          
            
            - `param_n_trials`: Number of trials to perform during the validation phase. 
            **Default: 100** 
            
            - `param_n_iterations`: Number of epochs to train the model on. 
            **Default: 1000**
            
            - `param_data_path`: Directory for output of spectrogram generation step. 
            **Default: `./data/images/`**
            
            - `param_dataset_path`: Directory for output of each run. 
            **Default: `./tmp`.**
            
            - `param_training_percentage`: Percentage of dataset for each gesture that will be allotted to the trainings set.
            **Default: 0.55.**
            
            - `param_wav_directory`: Directory of raw `.wav` files. 
            **Default: `./data/wav`**
    
    - `gen_spectrograms.py`: Produces spectrograms from raw sound data. 
    
    - `gen_datasets.py`: Prepare the dataset for the model training step. Divides the spectrogram image data generated by `gen_spectrograms.py` into training and validation data sets. Selects `param_training_percentage * #_samples_per_gesture` samples for the training dataset at random; the rest comprise the validation data set. 
    
    - `ml_siamese.py`: Defines the `Fewshot` implementation. Skeleton code taken from https://github.com/akshaysharma096/Siamese-Networks and heavily modified for the purposes of this assignment.  

    - `pipeline.py`: Runs the whole pipeline. 


Temporary folders
- `tmp` Stores the dataset, ML models, and results for each run 
    - `YYYY-MM-DD HH-MM-SS`: Overall run folder, corresponding to each time `pipeline.py` is run. 
        - `models`: Stores model weights
        - `results`: Stores results of training and validation, by type of spectrogram, as well as composite 

## Todo 

[ ] Add documentation for setting up the Matlab portion 

Download Matlab
Install Signal Processing kit () 
Follow instructions on https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html 

```
cd "matlabroot/extern/engines/python"
python setup.py install
```