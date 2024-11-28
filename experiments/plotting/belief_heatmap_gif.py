import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Load the CSV file
file_path = os.path.join(os.path.dirname(__file__), "../output/communication_vision/results.csv")
df = pd.read_csv("/Users/ciaran/dev/MultiAgentCommunication/experiments/output/communication_vision/results.csv")

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../output/communication_vision/plots")
os.makedirs(output_dir, exist_ok=True)

# Define grid layout
grid_layout = (4, 4)

# Normalize the belief values to ensure they are between 0 and 1
def normalize_beliefs(beliefs):
    beliefs_sum = [sum(row) for row in beliefs]
    normalized_beliefs = [[value / total if total != 0 else 0 for value in row] for row, total in zip(beliefs, beliefs_sum)]
    return normalized_beliefs

# Function to create animated heatmaps with enhanced layout and text
def create_heatmap_animation(agent_data, agent_id, other_agent_data, color_map="YlGnBu"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Adjust margins for a tighter layout
    plt.subplots_adjust(left=0.3, right=0.7, bottom=0.3, top=0.9)
    
    # Maximum timestep for filtering
    max_timestep = agent_data['timestep'].max()
    timestep_data = agent_data[agent_data['timestep'] <= max_timestep]
    
    # Custom x-axis and y-axis labels based on reward values and gem colors
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(['-5', '1', '2', '3'], fontsize=12)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['Yellow', 'Green', 'Blue', 'Red'], fontsize=12)
    
    # Static color bar for probability range
    sns.heatmap(np.zeros(grid_layout), vmin=0, vmax=1, cbar=True, annot=True, fmt=".2f", 
                cmap=color_map, square=True, ax=ax, cbar_kws={"shrink": 0.5})

    # Update function to add probability values only, with structured agent texts
    def update(row):
        ax.clear()
        
        # Probability matrix aligned with gem colors (rows) and rewards (columns)
        prob_matrix = np.array([
            [row['yellow_m5'], row['yellow_1'], row['yellow_2'], row['yellow_3']],
            [row['green_m5'], row['green_1'], row['green_2'], row['green_3']],
            [row['blue_m5'], row['blue_1'], row['blue_2'], row['blue_3']],
            [row['red_m5'], row['red_1'], row['red_2'], row['red_3']]
        ])
        
        # Plot heatmap with probability values in the chosen color map
        sns.heatmap(prob_matrix, vmin=0, vmax=1, annot=True, fmt=".2f", cmap=color_map, 
                    cbar=False, square=True, ax=ax)
        
        # Set correct x-axis and y-axis labels
        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(['-5', '1', '2', '3'], fontsize=12)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['Yellow', 'Green', 'Blue', 'Red'], fontsize=12)
        
        # Title and text structure for actions and utterances
        ax.set_title(f"Agent {agent_id} - Timestep {int(row['timestep'])}", fontsize=14)
        
        # Self-agent details on the left
        self_action = f"Self Agent Action: {row['pickup'].capitalize() if row['pickup'] != 'none' else 'No Pickup'}"
        self_utterance = f"Self Agent Utterance: {row['utterance']}" if row['utterance'] != 'none' else "Self Agent Utterance: None"
        
        # Other-agent details on the right (align by timestep)
        timestep = row['timestep']
        other_row = other_agent_data[other_agent_data['timestep'] == timestep].iloc[0]
        other_action = f"Other Agent Action: {other_row['pickup'].capitalize() if other_row['pickup'] != 'none' else 'No Pickup'}"
        other_utterance = f"Other Agent Utterance: {other_row['utterance']}" if other_row['utterance'] != 'none' else "Other Agent Utterance: None"
        
        # Display self-agent text on the left and other-agent text on the right, no bold
        ax.text(0.5, -0.2, self_action, ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(0.5, -0.3, self_utterance, ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.text(0.5, -0.4, other_action, ha='center', va='center', transform=ax.transAxes, fontsize=12,)
        ax.text(0.5, -0.5, other_utterance, ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # Create animation with frames spanning the timestep range
    anim = FuncAnimation(fig, update, frames=[row for _, row in timestep_data.iterrows()], repeat=False)
    filename = os.path.join(output_dir, f"agent_{agent_id}_heatmap_styled.gif")
    anim.save(filename, writer=PillowWriter(fps=2))
    plt.close()
    return filename

# Generate styled GIFs for each agent with full enhancements and save them
file_paths_styled = []
agents = df['agent'].unique()
for agent_id in agents:
    # Define self and other agent data
    agent_data = df[(df['repeat'] == 1) & (df['agent'] == agent_id)]
    other_agent_id = agents[1] if agent_id == agents[0] else agents[0]
    other_agent_data = df[(df['repeat'] == 1) & (df['agent'] == other_agent_id)]
    
    # Create GIF for current agent with full styling
    file_path = create_heatmap_animation(agent_data, agent_id, other_agent_data, color_map="YlGnBu")
    file_paths_styled.append(file_path)

print("Generated GIFs:", file_paths_styled)
