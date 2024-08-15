# Reinforcement Learning for Counterfactual Story Rewriting

This repository contains the code for training a model using reinforcement learning to rewrite story endings based on counterfactual events. The model is rewarded based on two custom metrics: Δ𝑀1 and Δ𝑀2, which measure how well the generated ending aligns with the edited ending and the counterfactual scenario, respectively.

## Task Overview
The task involves rewriting the endings of stories when a counterfactual event is introduced. The model is trained using reinforcement learning, where it interacts with an environment to generate story endings and receives rewards based on its performance against the Δ𝑀1 and Δ𝑀2 metrics.

### Story Components
Each story consists of the following structured components:
- **Premise (𝑋𝑃):** Sets the foundational scenario or context for the story.
- **Initial Event (𝑋𝐼𝐸):** Introduces an event that leads to the original story's conclusion.
- **Counterfactual Event (𝑋𝐶𝐸):** A divergent hypothetical event that alters the course of the story.
- **Original Ending (𝑋𝑂𝐸):** The original conclusion of the story.
- **Edited Ending (𝑌𝐸𝐸):** A modified ending that aligns with the counterfactual event.

### Custom Reward Metrics
The model's reward function is based on two metrics:
- **Δ𝑀1 = 𝑀 (Prediction, 𝑌𝐸𝐸 ) − 𝑀 (Prediction, 𝑋𝑂𝐸 )**: This metric measures how much better the generated ending (Prediction) aligns with the edited ending (𝑌𝐸𝐸) compared to the original ending (𝑋𝑂𝐸). A higher Δ𝑀1 score indicates that the generated narrative is more similar to the edited ending, which reflects the model’s ability to make the necessary changes required by the counterfactual event.

- **Δ𝑀2 = 𝑀 (Prediction, 𝑋𝐶𝐸 ) − 𝑀 (𝑌𝐸𝐸, 𝑋𝐶𝐸 )**: This metric measures how well the generated ending (Prediction) aligns with the counterfactual event (𝑋𝐶𝐸), normalized by the alignment between the edited ending (𝑌𝐸𝐸) and the counterfactual event. A higher Δ𝑀2 score indicates that the generated text aligns well with the counterfactual event, showing the model's ability to adapt the storyline logically given the counterfactual premise.

### Repository structure 
counterfactual-story-rewriting-rl/
├── data/                              # Directory to store the dataset files
│   ├── train.json                     # Training dataset
│   ├── dev.json                       # Development/Validation dataset
│   └── test.json                      # Test dataset
├── src/                               # Main source code directory
│   ├── __init__.py                    # Initialization file for the src module
│   ├── config.py                      # Configuration settings for the project
│   ├── data_loader.py                 # Script for loading and preprocessing data
│   ├── main_rl.py                     # Main script to run the RL training loop
│   ├── environment.py                 # Script defining the RL environment, including state, action, and reward functions
│   └── utils/                         # Utility scripts including reward calculations and preprocessing
│       ├── __init__.py                # Initialization file for utils
│       ├── rewards.py                 # Functions to calculate Δ𝑀1, Δ𝑀2, and similarity
│       └── preprocess.py              # Preprocessing functions for the dataset
├── results/                           # Directory to save the results of the training (e.g., models, generated outputs)
├── logs/                              # Directory to save training logs (e.g., TensorBoard logs)
├── requirements.txt                   # List of dependencies required for the project
├── README.md                          # Project README file
└── LICENSE                            # License file for the project



### Usage

#### Training the Model
To train the model using reinforcement learning:

1. Clone the repository.
2. Install the necessary dependencies.
3. Run the training script.

```bash
git clone https://github.com/yourusername/counterfactual-story-rewriting-rl.git
cd counterfactual-story-rewriting-rl
pip install -r requirements.txt
python main_rl.py
```

