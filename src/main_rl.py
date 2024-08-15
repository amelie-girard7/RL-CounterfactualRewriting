import sys
from pathlib import Path
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from pytorch_lightning import Trainer

# Ensure the script can find the 'src' module
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / 'src'
sys.path.append(str(src_dir))

from config import CONFIG
from environment import StoryEnvironment
from data_loader import create_datasets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model directory and checkpoint paths
model_directory_path = "/data/agirard/Projects/RL-CounterfactualRewriting/models/model_2024-03-22-10"
save_directory = "/data/agirard/Projects/RL-CounterfactualRewriting/models"

# Load the pre-trained model from Hugging Face
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def generate_ending(state):
    """
    Generates a story ending based on the state using the pre-trained model.
    """
    # Prepare the input for the model (concatenate state information)
    input_text = f"{state['premise']} {state['initial']} {state['counterfactual']} {state['original_ending']}"
    
    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=CONFIG["max_length"], truncation=True)
    
    # Generate the ending
    with torch.no_grad():  # Disable gradient calculation for inference
        output_ids = model.generate(input_ids, max_length=CONFIG["max_gen_length"])
    
    # Decode the output
    generated_ending = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_ending

def main():
    # Load data
    train_data, dev_data, test_data = create_datasets(CONFIG)

    # Initialize environment
    env = StoryEnvironment(train_data)

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(max_epochs=CONFIG["max_epochs"])

    # Track cumulative reward over all episodes
    total_reward = 0

    # Run training for a number of episodes
    for episode in range(1, CONFIG["max_epochs"] + 1):
        state = env.reset()  # Reset environment to start a new episode
        action = generate_ending(state)  # Generate an ending based on the current state

        # Take an action in the environment (generate an ending)
        reward, next_state, done = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Log the details of this episode
        logger.info(f"Episode {episode}: Generated Ending: {action} | Reward: {reward}")

    # Log the total reward after all episodes
    logger.info(f"Total cumulative reward after {CONFIG['max_epochs']} episodes: {total_reward}")

    # Save the trained model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    logger.info(f"Model and tokenizer saved to {save_directory}")

if __name__ == '__main__':
    main()
