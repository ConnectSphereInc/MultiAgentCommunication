import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Load the CSV file
file_path = os.path.join(os.path.dirname(__file__), "../output/results.csv")
df = pd.read_csv("/Users/ciaran/dev/MultiAgentCommunication/debugging/output/results.csv")

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "../output/plots")
os.makedirs(output_dir, exist_ok=True)

# Define grid layout
grid_layout = (4, 4)

def create_dual_heatmap_animation(agent1_data, agent2_data, color_map="YlGnBu"):
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(16, 12))  # Increased overall figure height
    
    # Create a specific layout with gridspec
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1])  # 2x2 grid with more space for heatmaps
    ax1 = fig.add_subplot(gs[0, 0])  # First heatmap
    ax2 = fig.add_subplot(gs[0, 1])  # Second heatmap
    text_ax = fig.add_subplot(gs[1, :])  # Full width for text
    text_ax.axis('off')  # Hide the text axis
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
    
    # Setup axes labels and ticks for both plots
    for ax in [ax1, ax2]:
        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(['-5', '1', '2', '3'], fontsize=12)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['Yellow', 'Green', 'Blue', 'Red'], fontsize=12)
    
    # Initial heatmaps with zero matrices for colorbar setup
    sns.heatmap(np.zeros(grid_layout), vmin=0, vmax=1, cbar=True, annot=True, fmt=".2f",
                cmap=color_map, square=True, ax=ax1, cbar_kws={"shrink": 0.5})
    sns.heatmap(np.zeros(grid_layout), vmin=0, vmax=1, cbar=True, annot=True, fmt=".2f",
                cmap=color_map, square=True, ax=ax2, cbar_kws={"shrink": 0.5})

    def update(frame):
        timestep = frame
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        text_ax.clear()
        text_ax.axis('off')
        
        # Get data for current timestep
        agent1_row = agent1_data[agent1_data['timestep'] == timestep].iloc[0]
        agent2_row = agent2_data[agent2_data['timestep'] == timestep].iloc[0]
        
        # Create probability matrices for both agents
        def create_prob_matrix(row):
            return np.array([
                [row['yellow_m5'], row['yellow_1'], row['yellow_2'], row['yellow_3']],
                [row['green_m5'], row['green_1'], row['green_2'], row['green_3']],
                [row['blue_m5'], row['blue_1'], row['blue_2'], row['blue_3']],
                [row['red_m5'], row['red_1'], row['red_2'], row['red_3']]
            ])
        
        prob_matrix1 = create_prob_matrix(agent1_row)
        prob_matrix2 = create_prob_matrix(agent2_row)
        
        # Plot heatmaps
        sns.heatmap(prob_matrix1, vmin=0, vmax=1, annot=True, fmt=".2f", cmap=color_map,
                    cbar=False, square=True, ax=ax1)
        sns.heatmap(prob_matrix2, vmin=0, vmax=1, annot=True, fmt=".2f", cmap=color_map,
                    cbar=False, square=True, ax=ax2)
        
        # Reset ticks and labels for both plots
        for ax in [ax1, ax2]:
            ax.set_xticks([0.5, 1.5, 2.5, 3.5])
            ax.set_xticklabels(['-5', '1', '2', '3'], fontsize=12)
            ax.set_yticks([0.5, 1.5, 2.5, 3.5])
            ax.set_yticklabels(['Yellow', 'Green', 'Blue', 'Red'], fontsize=12)
        
        # Set titles
        ax1.set_title(f"Agent {agent1_row['agent']} - Timestep {int(timestep)}", fontsize=14)
        ax2.set_title(f"Agent {agent2_row['agent']} - Timestep {int(timestep)}", fontsize=14)
        
        # Format action and utterance text
        def format_agent_text(row, agent_num):
            action = f"Agent {agent_num} Action: {row['pickup'].capitalize() if row['pickup'] != 'none' else 'No Pickup'}"
            reward = f"Agent {agent_num} Observed Reward: {row['observed_reward']} | Ground Truth Reward: {row['ground_truth_reward']}"
            utterance = f"Agent {agent_num} Utterance: {row['utterance']}" if row['utterance'] != 'none' else f"Agent {agent_num} Utterance: None"
            return action, reward, utterance
        
        # Get formatted text for both agents
        agent1_action, agent1_observed_reward, agent1_utterance = format_agent_text(agent1_row, 1)
        agent2_action, agent2_observed_reward, agent2_utterance = format_agent_text(agent2_row, 2)
        
        # Add text using the text axis
        text = f"{agent1_action}\n{agent1_observed_reward}\n{agent1_utterance}\n{agent2_observed_reward}\n{agent2_action}\n{agent2_utterance}"
        text_ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, 
                    transform=text_ax.transAxes, linespacing=2)

    # Create animation
    unique_timesteps = sorted(pd.concat([agent1_data['timestep'], agent2_data['timestep']]).unique())
    anim = FuncAnimation(fig, update, frames=unique_timesteps, repeat=False)
    
    # Save animation
    filename = os.path.join(output_dir, "dual_agent_heatmap.gif")
    anim.save(filename, writer=PillowWriter(fps=2))
    plt.close()
    return filename

# Generate the dual agent animation
agents = df['agent'].unique()
agent1_data = df[df['agent'] == agents[0]]
agent2_data = df[df['agent'] == agents[1]]

file_path = create_dual_heatmap_animation(agent1_data, agent2_data, color_map="YlGnBu")
print("Generated dual agent GIF:", file_path)