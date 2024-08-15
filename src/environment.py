import random
from utils.rewards import calculate_delta_m1, calculate_delta_m2

class StoryEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_index = 0

    def reset(self):
        self.current_index = random.randint(0, len(self.data) - 1)
        return self.get_state()

    def get_state(self):
        story = self.data[self.current_index]
        return {
            'premise': story['premise'],
            'initial': story['initial'],
            'counterfactual': story['counterfactual'],
            'original_ending': story['original_ending'],
            'edited_ending': story['edited_ending']
        }

    def step(self, action):
        story = self.data[self.current_index]
        generated_ending = action
        delta_m1 = calculate_delta_m1(generated_ending, story)
        delta_m2 = calculate_delta_m2(generated_ending, story)
        reward = delta_m1 + delta_m2
        return reward, self.get_state(), self.is_done()

    def is_done(self):
        return True  # Each episode is a single step for simplicity
