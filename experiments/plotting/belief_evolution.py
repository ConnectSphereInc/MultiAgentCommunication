import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV file
file_path = os.path.join(os.path.dirname(__file__), "../output/communication_vision/results.csv")
df = pd.read_csv(file_path)

def plot_most_likely_gem_values(problem_name):
    # Filter data for the specific problem
    problem_data = df[df['problem_name'] == problem_name]

    # Filter data for agent 1 within the specific problem
    agent_1_data = problem_data[problem_data['agent'] == 1].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Function to find the most likely value for each timestep for a specific gem color
    def most_likely_values(agent_data, gem_prefix):
        cols = [f'{gem_prefix}_m5', f'{gem_prefix}_1', f'{gem_prefix}_2', f'{gem_prefix}_3']
        values = np.array([-5, 1, 2, 3])
        return values[np.argmax(agent_data[cols].values, axis=1)]

    # Calculate most likely gem value for each timestep for all gem colors
    agent_1_data.loc[:, 'most_likely_red'] = most_likely_values(agent_1_data, 'red')
    agent_1_data.loc[:, 'most_likely_blue'] = most_likely_values(agent_1_data, 'blue')
    agent_1_data.loc[:, 'most_likely_green'] = most_likely_values(agent_1_data, 'green')
    agent_1_data.loc[:, 'most_likely_yellow'] = most_likely_values(agent_1_data, 'yellow')

    # Map the gem values to discrete numbers for categorical plotting
    value_map = {-5: 0, 1: 1, 2: 2, 3: 3}

    # Apply the mapping to the most likely gem values for easier plotting
    agent_1_data.loc[:, 'most_likely_red_mapped'] = agent_1_data['most_likely_red'].map(value_map)
    agent_1_data.loc[:, 'most_likely_blue_mapped'] = agent_1_data['most_likely_blue'].map(value_map)
    agent_1_data.loc[:, 'most_likely_green_mapped'] = agent_1_data['most_likely_green'].map(value_map)
    agent_1_data.loc[:, 'most_likely_yellow_mapped'] = agent_1_data['most_likely_yellow'].map(value_map)

    # Filter data for agent 2 (to get utterances)
    agent_2_data = problem_data[problem_data['agent'] == 2]

    # Plot the evolution of most likely gem values with adjusted colors and opacity
    plt.figure(figsize=(10, 6))

    plt.plot(agent_1_data['timestep'], agent_1_data['most_likely_red_mapped'], label='Red Gem', color='crimson', alpha=0.8)
    plt.plot(agent_1_data['timestep'], agent_1_data['most_likely_blue_mapped'], label='Blue Gem', color='royalblue', alpha=0.8)
    plt.plot(agent_1_data['timestep'], agent_1_data['most_likely_green_mapped'], label='Green Gem', color='forestgreen', alpha=0.8)
    plt.plot(agent_1_data['timestep'], agent_1_data['most_likely_yellow_mapped'], label='Yellow Gem', color='gold', alpha=0.8)

    # Add vertical lines for agent 1 pickups
    for t in agent_1_data[agent_1_data['pickup'] != 'none']['timestep']:
        plt.axvline(x=t, color='black', linestyle='-', alpha=0.6, label='Agent 1 Pickup' if t == agent_1_data[agent_1_data['pickup'] != 'none']['timestep'].iloc[0] else "")

    # Add vertical lines for Agent 2's communication in the previous timestep
    for t in agent_2_data[agent_2_data['utterance'] != 'none']['timestep']:
        plt.axvline(x=t + 1, color='black', linestyle='--', alpha=0.5, label='Agent 2 Communication' if t == agent_2_data[agent_2_data['utterance'] != 'none']['timestep'].iloc[0] else "")

    # Adjust y-ticks to reflect the reward values
    plt.yticks(list(value_map.values()), list(value_map.keys()))

    plt.xlabel('Timestep')
    plt.ylabel('Most Likely Gem Reward')
    plt.title(f"Most Likely Gem Value for Agent 1 Over Time for {problem_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "../output/communication_vision/plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{problem_name.replace("/", "-")}_belief_evolution.png'), dpi=300)

# Example usage:
plot_most_likely_gem_values('medium/1')
