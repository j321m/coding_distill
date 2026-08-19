# experiments/basic.py

from munch import Munch
from mrunner.experiment import Experiment

experiment = Experiment(
    name="toy_exp",
    script="pixi run python main.py",
    parameters=Munch(
        learning_rate=0.001,
        batch_size=64,
    ),
)

experiments_list = [experiment]
