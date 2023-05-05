# DDPG-Her-DecisionTree-vs.-PID
This repository contains the code to compare two control approaches for the Panda Reach v2 environment provided by OpenAI Gym: Deep Deterministic Policy Gradients with Hindsight Experience Replay and Decision Tree regression (DDPG+HER+DecisionTree), and Proportional-Integral-Derivative (PID) controller.

##Environment setup
Conda environment

First, create a Conda environment with the required dependencies:

    conda env create -f environment.yml

Installing the environment

Next, activate the environment:

    conda activate safeRL

Running the environment

To run the environment with DDPG+HER+DecisionTree controller:



    python DDPG_Reach.py

To run the environment with PID controller:



    python PID_Reach.py

##Results

After running the scripts, the results will be saved in the logs directory. The Tensorboard files can be visualized with:


    tensorboard --logdir=logs

##DDPG+HER+DecisionTree controller

After training, the DDPG+HER+DecisionTree controller will be saved in the models/DDPG_Reach directory.
PID controller

The PID controller results will be saved in the logs/PID_Reach directory.
Conclusion

Comparing the performance of DDPG+HER+DecisionTree and PID controllers can be done by comparing the respective Tensorboard logs. The metrics for comparison are rollout/ep_len_mean (episode length mean), rollout/ep_rew_mean (episode reward mean), and rollout/success_rate (success rate).

Based on the comparison, we can draw conclusions about the effectiveness of the two controllers for the Panda Reach v2 environment.

##Acknowledgements

OpenAI for providing the Panda Reach v2 environment

Stable Baselines 3 for the DDPG implementation

