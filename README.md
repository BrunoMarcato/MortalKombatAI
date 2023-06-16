# Mortal Kombat AI

This repository implements a way to train an agent to play the 1992 classic Mortal kombat.

## Rom

You must import the rom using the command below, inside the folder where it is located, so that it is recognized

`python -m retro.import .`

## Run

Please, run train.py file via terminal, after import the rom, passing the total_timesteps number, of model.learn function, as an argument.

`python -m train.py [total_timesteps]`

## Notes

The requirements.txt file lists all the dependencies that is needed to run this project. They will be installed using:

`pip install -r requirements.txt`
