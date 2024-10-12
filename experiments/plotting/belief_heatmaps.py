import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file
file_path = os.path.join(os.path.dirname(__file__), "../output/communication_vision/results.csv")
df = pd.read_csv(file_path)

# Normalize the belief values to ensure they are between 0 and 1
def normalize_beliefs(beliefs):
    beliefs_sum = [sum(row) for row in beliefs]
    normalized_beliefs = [[value / total if total != 0 else 0 for value in row] for row, total in zip(beliefs, beliefs_sum)]
    return normalized_beliefs

def plot_heatmaps_for_agents_separate_large_font(problem_name, timestep):
    # Filter data for the specified problem and timestep
    problem_data = df[(df['problem_name'] == problem_name) & (df['timestep'] == timestep)]
    
    # Get data for agent 1
    agent_1_data = problem_data[problem_data['agent'] == 1].copy()
    
    # Check if agent 1 had an utterance and didn't pick up a gem
    if agent_1_data.iloc[0]['pickup'] == 'none':
        agent_1_next_timestep = df[(df['problem_name'] == problem_name) & (df['timestep'] == timestep + 1) & (df['agent'] == 1)].copy()
    else:
        agent_1_next_timestep = agent_1_data
    
    # Get data for agent 2 at the same timestep
    agent_2_data = problem_data[problem_data['agent'] == 2].copy()
    
    # Check if agent 2 had an utterance and didn't pick up a gem
    if agent_2_data.iloc[0]['pickup'] == 'none':
        agent_2_next_timestep = df[(df['problem_name'] == problem_name) & (df['timestep'] == timestep + 1) & (df['agent'] == 2)].copy()
    else:
        agent_2_next_timestep = agent_2_data
    
    # Prepare heatmap data for agent 1 and agent 2
    rewards = [-5, 1, 2, 3]
    gems = ['red', 'blue', 'green', 'yellow']

    agent_1_beliefs = [
        [agent_1_next_timestep[f'{gem}_m5'].values[0], agent_1_next_timestep[f'{gem}_1'].values[0], 
         agent_1_next_timestep[f'{gem}_2'].values[0], agent_1_next_timestep[f'{gem}_3'].values[0]] 
        for gem in gems
    ]

    agent_2_beliefs = [
        [agent_2_next_timestep[f'{gem}_m5'].values[0], agent_2_next_timestep[f'{gem}_1'].values[0], 
         agent_2_next_timestep[f'{gem}_2'].values[0], agent_2_next_timestep[f'{gem}_3'].values[0]] 
        for gem in gems
    ]
    
    # Normalize beliefs for agent 1 and agent 2
    agent_1_beliefs_normalized = normalize_beliefs(agent_1_beliefs)
    agent_2_beliefs_normalized = normalize_beliefs(agent_2_beliefs)

    # Set font size
    font_size = 14

    # Plot heatmap for agent 1
    plt.figure(figsize=(6, 6))
    sns.heatmap(agent_1_beliefs_normalized, annot=True, cmap="YlGnBu", yticklabels=rewards, xticklabels=gems, vmin=0, vmax=1, annot_kws={"size": font_size})
    plt.title(f'Agent 1 Beliefs, P(Rewards | Observations)', fontsize=font_size)
    plt.ylabel('Reward', fontsize=font_size)
    plt.xlabel('Gem', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "../output/communication_vision/plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot for agent 1
    output_path = os.path.join(output_dir, f'{problem_name.replace("/", "-")}_heatmap_timestep_{timestep}_agent_1.png')
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Plot heatmap for agent 2
    plt.figure(figsize=(6, 6))
    sns.heatmap(agent_2_beliefs_normalized, annot=True, cmap="YlGnBu", yticklabels=rewards, xticklabels=gems, vmin=0, vmax=1, annot_kws={"size": font_size})
    plt.title(f'Agent 2 Beliefs, P(Rewards | Observations)', fontsize=font_size)
    plt.ylabel('Reward', fontsize=font_size)
    plt.xlabel('Gem', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    # Save the plot for agent 2
    output_path = os.path.join(output_dir, f'{problem_name.replace("/", "-")}_heatmap_timestep_{timestep}_agent_2.png')
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close()

# Example usage:
plot_heatmaps_for_agents_separate_large_font('medium/1', 8)
