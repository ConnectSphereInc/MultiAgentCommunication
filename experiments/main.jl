using MultiAgentCommunication
using DotEnv
using PDDL

PDDL.Arrays.register!()

function main()
    overlay = DotEnv.config()
    api_key = get(overlay, "OPENAI_API_KEY", nothing)
    ENV["OPENAI_API_KEY"] = api_key

    problem_name = "medium"
    ess_thresh = 0.3
    num_particles = 50
    ground_truth_rewards = Dict(:red => 1, :blue => -5, :yellow => 3, :green => 2)
    T = 100
    
    # Define the output directory
    output_dir = joinpath(@__DIR__, "output")

    _ = run_simulation_communication_vision(
        problem_name,
        ess_thresh,
        num_particles,
        ground_truth_rewards,
        T,
        output_dir
    )

    _ = run_simulation_no_communication_vision(
        problem_name,
        ess_thresh,
        num_particles,
        ground_truth_rewards,
        T,
        output_dir
    )

    _ = run_simulation_communication_restricted_vision(
        problem_name,
        ess_thresh,
        num_particles,
        ground_truth_rewards,
        T,
        output_dir
    )

    _ = run_simulation_communication_perfect_vision(
        problem_name,
        ess_thresh,
        num_particles,
        ground_truth_rewards,
        T,
        output_dir
    )

    _ = run_simulation_gpt4o(
        problem_name,
        ground_truth_rewards,
        T,
        output_dir
    )

end

main()