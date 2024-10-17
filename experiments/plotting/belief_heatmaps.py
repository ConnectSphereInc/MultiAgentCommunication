import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = os.path.join(os.path.dirname(__file__), "../output/communication_vision/results.csv")
df = pd.read_csv(file_path)


# Normalize the belief values to ensure they are between 0 and 1
def normalize_beliefs(beliefs):
    beliefs_sum = [sum(row) for row in beliefs]
    normalized_beliefs = [[value / total if total != 0 else 0 for value in row] for row, total in zip(beliefs, beliefs_sum)]
    return normalized_beliefs

def plot_heatmaps_for_agents_at_t_plus_1(df, problem_name, timestep, repeat):
    
    next_timestep = timestep + 1

    # Filter data for the specified problem, repeat, and timestep + 1
    filtered_df = df[(df['problem_name'] == problem_name) & 
                     (df['timestep'] == next_timestep) & 
                     (df['repeat'] == repeat)]

    if filtered_df.empty:
        print(f"No data available for problem '{problem_name}', timestep '{next_timestep}', and repeat '{repeat}'")
        return

    # Extract belief data for agent 1 and agent 2
    agent_1_data = filtered_df[filtered_df['agent'] == 1].iloc[0]
    agent_2_data = filtered_df[filtered_df['agent'] == 2].iloc[0]

    # Create a list of gem colors and rewards
    gems = ['red', 'blue', 'green', 'yellow']
    rewards = ['-5', '1', '2', '3']  # Change m5 to -5

    # Extract the belief values for each gem for both agents
    agent_1_beliefs = [
        [agent_1_data[f'red_{reward.replace("-5", "m5")}'] for reward in rewards],
        [agent_1_data[f'blue_{reward.replace("-5", "m5")}'] for reward in rewards],
        [agent_1_data[f'green_{reward.replace("-5", "m5")}'] for reward in rewards],
        [agent_1_data[f'yellow_{reward.replace("-5", "m5")}'] for reward in rewards]
    ]
    agent_2_beliefs = [
        [agent_2_data[f'red_{reward.replace("-5", "m5")}'] for reward in rewards],
        [agent_2_data[f'blue_{reward.replace("-5", "m5")}'] for reward in rewards],
        [agent_2_data[f'green_{reward.replace("-5", "m5")}'] for reward in rewards],
        [agent_2_data[f'yellow_{reward.replace("-5", "m5")}'] for reward in rewards]
    ]

    # Normalize the belief values
    agent_1_beliefs_normalized = normalize_beliefs(agent_1_beliefs)
    agent_2_beliefs_normalized = normalize_beliefs(agent_2_beliefs)

    font_size = 14

    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "../output/communication_vision/plots")
    os.makedirs(output_dir, exist_ok=True)

    # Plot heatmap for agent 1 (switching gem and reward axes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(agent_1_beliefs_normalized, annot=True, cmap="YlGnBu", xticklabels=rewards, yticklabels=gems, vmin=0, vmax=1, 
                annot_kws={"size": font_size}, fmt=".2f")  # Format to 2 decimal places
    plt.title(f'Agent 1 Beliefs, P(Rewards | Observations)', fontsize=font_size)
    plt.ylabel('Gem', fontsize=font_size)
    plt.xlabel('Reward', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    # Save the plot for agent 1
    output_path_agent_1 = os.path.join(output_dir, f'{problem_name.replace("/", "-")}_heatmap_timestep_{next_timestep}_agent_1_repeat_{repeat}.png')
    print(f"Saving plot to {output_path_agent_1}")
    plt.savefig(output_path_agent_1, dpi=300)
    plt.close()

    # Plot heatmap for agent 2 (switching gem and reward axes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(agent_2_beliefs_normalized, annot=True, cmap="YlGnBu", xticklabels=rewards, yticklabels=gems, vmin=0, vmax=1, 
                annot_kws={"size": font_size}, fmt=".2f")  # Format to 2 decimal places
    plt.title(f'Agent 2 Beliefs, P(Rewards | Observations)', fontsize=font_size)
    plt.ylabel('Gem', fontsize=font_size)
    plt.xlabel('Reward', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    # Save the plot for agent 2
    output_path_agent_2 = os.path.join(output_dir, f'{problem_name.replace("/", "-")}_heatmap_timestep_{next_timestep}_agent_2_repeat_{repeat}.png')
    print(f"Saving plot to {output_path_agent_2}")
    plt.savefig(output_path_agent_2, dpi=300)
    plt.close()

# Example usage with repeat=1
plot_heatmaps_for_agents_at_t_plus_1(df, 'medium/1', 6, 5)
plot_heatmaps_for_agents_at_t_plus_1(df, 'medium/1', 8, 5)
plot_heatmaps_for_agents_at_t_plus_1(df, 'medium/1', 9, 5)
plot_heatmaps_for_agents_at_t_plus_1(df, 'medium/1', 99, 5)