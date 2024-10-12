import os
import pandas as pd

# Define the directories that contain the result CSVs, including gpt4o
output_folders = [
    'communication_vision',
    'no_communication_vision',
    'communication_restricted_vision',
    'communication_perfect_vision',
    'gpt4o'  # Added gpt4o folder
]

# Define the base path where the folders are located (adjust this as needed)
base_path = 'experiments/output'

# Ground truth rewards
ground_truth_rewards = {
    'red': 1,
    'blue': -5,
    'yellow': 3,
    'green': 2
}

# List to store the data for the final CSV
average_rewards_data = []

# Helper function to infer the agent's belief based on the highest probability
def infer_belief(df, color):
    prob_columns = [f"{color}_m5", f"{color}_1", f"{color}_2", f"{color}_3"]
    most_likely_reward_index = df[prob_columns].iloc[-1].idxmax()  # Get the index of the max probability
    if most_likely_reward_index.endswith('m5'):
        return -5
    else:
        return int(most_likely_reward_index.split('_')[-1])  # Extract the reward from the column name

# Loop over each folder
for folder in output_folders:
    # Construct the path to the results.csv file
    file_path = os.path.join(base_path, folder, 'results.csv')
    
    # Load the CSV file
    if not os.path.exists(file_path):
        print(f"File not found for folder {folder}, skipping...")
        continue

    data = pd.read_csv(file_path)
    
    # Check if this is a gpt4o results file based on the columns
    is_gpt4o = 'red' in data.columns and 'blue' in data.columns and 'green' in data.columns and 'yellow' in data.columns
    
    # Group by problem_name and repeat, get the last timestep's score for each group, sum both agents' scores
    grouped_data = data.groupby(['problem_name', 'repeat'])
    final_scores = grouped_data.apply(lambda df: df[df['timestep'] == df['timestep'].max()]['score'])
    
    # Calculate the average final score and the standard error (uncertainty)
    average_final_score = final_scores.mean()
    std_dev = final_scores.std()
    std_error = std_dev / (len(final_scores) ** 0.5)  # Standard error
    
    # Calculate the percentage of correct inferences
    correct_inferences = 0
    total_inferences = 0
    
    for _, group in grouped_data:
        for agent in group['agent'].unique():
            agent_data = group[group['agent'] == agent]
            
            if is_gpt4o:
                # For gpt4o, the values are directly the agents' beliefs
                inferred_beliefs = {
                    'red': agent_data['red'].iloc[-1],
                    'blue': agent_data['blue'].iloc[-1],
                    'yellow': agent_data['yellow'].iloc[-1],
                    'green': agent_data['green'].iloc[-1]
                }
            else:
                # For other types, infer the agent's belief for each gem color
                inferred_beliefs = {
                    'red': infer_belief(agent_data, 'red'),
                    'blue': infer_belief(agent_data, 'blue'),
                    'yellow': infer_belief(agent_data, 'yellow'),
                    'green': infer_belief(agent_data, 'green')
                }
            
            # Compare beliefs with ground truth
            for color, ground_truth in ground_truth_rewards.items():
                if inferred_beliefs[color] == ground_truth:
                    correct_inferences += 1
                total_inferences += 1
    
    # Calculate the percentage of correct inferences
    percent_correct_inferences = (correct_inferences / total_inferences) * 100 if total_inferences > 0 else 0
    
    # Append the type (folder), average reward, uncertainty, and percentage correct inferences to the list
    average_rewards_data.append({
        'type': folder,
        'average_reward': average_final_score,
        'std_error': std_error,
        'percent_correct_inferences': percent_correct_inferences
    })

# Create a DataFrame from the collected data
average_rewards_df = pd.DataFrame(average_rewards_data)

# Save the DataFrame to a CSV file
output_file_path = 'experiments/analysis/analysis.csv'
average_rewards_df.to_csv(output_file_path, index=False)

print("Average rewards with uncertainties and correctness have been saved to:", output_file_path)
