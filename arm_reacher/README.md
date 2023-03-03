[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Arm Reacher

### Project Details

The environment contains a double-jointed robotic arm (with 2 degree of freedom). 
The task is to get the end-effector of the robot arm to track the moving target (the green sphere) as accurately as possible. 

We use the Unity ML-Agents environment. This is performed as part of a project for the course Deep Reinforcement Learning Nanodegree.
See example below. 

![Trained Agent][image1]

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of our agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
Every entry in the action vector should be a number between -1 and 1.

The task is set to be episodic where a user-defined timeout is provided.

In this project, we train with **20 agents** in parallel. 
Let's say the average total non-discounted returns of all 20 agents is what we call a "score". 

The agent is judged with the average score over 100 consecutive episodes. 
To consider the task solved, we set a goal of average score >= 30 (over 100 consecutive episodes).

### Getting Started

1. Set up a conda virtual environment.

```bash 
conda create --name arm_reacher python=3.6 
conda activate arm_reacher 
```

2. Clone the repository. 

```bash 
git clone git@github.com:PrajishKumar/Reinforcement-Learning-Unity-ML-Engine.git
```

3. Install all the requirements for this project.

```bash
cd arm_reacher/python
pip install . 
cd ../code
```

4. Create an IPython kernel for the environment. 

```bash
python -m ipykernel install --user --name arm_reacher --display-name "arm_reacher"
```

5. Open the notebook. 

```bash
jupyter notebook ArmReacher.ipynb
```

6. Before running code in a notebook, change the kernel to match the `arm_reacher` environment. 
In the menu bar, choose `Kernel` -> `Change kernel` -> `arm_reacher`.

### Instructions

Follow the instructions in `code/ArmReacher.ipynb` to see the agent learn to track the moving target in play.
The instructions in the notebook should be sufficiently clear to guide you.