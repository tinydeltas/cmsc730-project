#!/bin/zsh

python3 -m venv venv 
source venv/bin/activate

python3 -m pip install jupyter
ipython kernel install --name "local-venv" --user
python3 -m pip install -r requirements.txt
