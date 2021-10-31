#!/bin/zsh
python3 -m venv venv 
source venv/bin/activate
pip install jupyter
ipython kernel install --name "local-venv" --user
