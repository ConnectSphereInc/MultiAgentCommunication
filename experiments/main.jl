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
    num_particles = 200
    ground_truth_rewards = Dict(:red => 1, :blue => -5, :yellow => 3, :green => 2)
    T = 50
    
    # Define the output directory
    output_dir = joinpath(@__DIR__, "output")

    final_score = run_simulation(
        problem_name,
        ess_thresh,
        num_particles,
        ground_truth_rewards,
        T,
        output_dir
    )

    println("\nFinal Score: $final_score\n")
end

main()