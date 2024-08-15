# Reinforcement Learning for Counterfactual Story Rewriting
This repository leverages reinforcement learning to generate coherent story endings based on counterfactual scenarios. It features a custom metrics (Δ𝑀1, Δ𝑀2) identified in the paper "Training Objectives and Evaluation Metrics for Counterfactual Story Rewriting" for targeted reward functions, fine-tunes a T5 model, and offers a streamlined codebase for efficient training and evaluation.

## Task Overview
The task involves rewriting the endings of stories when a counterfactual event is introduced. The model is trained using reinforcement learning, where it interacts with an environment to generate story endings and receives rewards based on its performance against custom metrics.

## Story Components
Each story consists of the following structured components:

- **Premise (𝑋𝑃):** Sets the foundational scenario or context for the story.
- **Initial Event (𝑋𝐼𝐸):** Introduces an event that leads to the original story's conclusion.
- **Original Ending (𝑋𝑂𝐸):** The original conclusion of the story.
- **Counterfactual Event (𝑋𝐶𝐸):** A divergent hypothetical event that alters the course of the story.

## Custom Reward Function
The model's reward function is based on two custom metrics designed to evaluate the quality of the generated story endings:

### Δ𝑀1

This metric measures how much better the model's prediction aligns with the edited ending compared to the original ending. The equation is as follows:

\[
\Delta M_1 = M(\text{Prediction}, Y_{\text{EE}}) - M(\text{Prediction}, X_{\text{OE}})
\]

Where:
- \( M \) is a conventional metric (e.g., ROUGE, BERTScore) used to assess the similarity between the prediction and the reference.
- \( Y_{\text{EE}} \) is the edited ending.
- \( X_{\text{OE}} \) is the original ending.

### Δ𝑀2

This metric evaluates how well the model's prediction aligns with the counterfactual event compared to the edited ending. The equation is as follows:

\[
\Delta M_2 = M(\text{Prediction}, X_{\text{CE}}) - M(Y_{\text{EE}}, X_{\text{CE}})
\]

Where:
- \( M \) is a conventional metric used to assess the similarity between the prediction and the reference.
- \( X_{\text{CE}} \) is the counterfactual event.
- \( Y_{\text{EE}} \) is the edited ending.

## Usage

### Training the Model
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


## Project Structure

The project structure will now be minimal and focused:

```plaintext
counterfactual-story-rewriting-rl/
├── models/                  # Directory for storing trained models or model-related files.
├── src/                     # Main source code directory.
│   ├── models/              # Model-related scripts, including model_T5.py.
│   ├── utils/               # Utility scripts including config, utils, and data_loader.
│   │   ├── config.py        # Configuration settings for the project.
│   │   ├── utils.py         # Essential utility functions.
│   │   └── data_loader.py   # Data loading and preprocessing for RL.
├── main_rl.py               # Main script to run the RL training.
├── README.md                # Project README file.
└── requirements.txt         # List of dependencies required for the project.
```

