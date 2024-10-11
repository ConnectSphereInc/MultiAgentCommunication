using MultiAgentCommunication
using DotEnv
using PDDL

PDDL.Arrays.register!()

function main()
    overlay = DotEnv.config()
    api_key = get(overlay, "OPENAI_API_KEY", nothing)
    ENV["OPENAI_API_KEY"] = api_key

    ess_thresh = 0.3
    num_particles = 10
    ground_truth_rewards = Dict(:red => 1, :blue => -5, :yellow => 3, :green => 2)
    T = 100
    
    problem_folder = joinpath(@__DIR__, "..", "src/problems/medium")
    files = readdir(problem_folder)
    problems = [joinpath(problem_folder, file) for file in files]
    
    task = "communication_vision"
    for problem in problems
        name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
        output_dir = joinpath(@__DIR__, "output", task, name)
        mkpath(output_dir)
        _ = run_simulation_communication_vision(
            problem,
            output_dir,
            ess_thresh,
            num_particles,
            ground_truth_rewards,
            T
        )
    end

    task = "no_communication_vision"
    for problem in problems
        name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
        output_dir = joinpath(@__DIR__, "output", task, name)
        mkpath(output_dir)
        _ = run_simulation_no_communication_vision(
            problem,
            output_dir,
            ess_thresh,
            num_particles,
            ground_truth_rewards,
            T
        )
    end

    task = "communication_restricted_vision"
    for problem in problems
        name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
        output_dir = joinpath(@__DIR__, "output", task, name)
        mkpath(output_dir)
        _ = run_simulation_communication_restricted_vision(
            problem,
            output_dir,
            ess_thresh,
            num_particles,
            ground_truth_rewards,
            T
        )
    end

    task = "communication_perfect_vision"
    for problem in problems
        name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
        output_dir = joinpath(@__DIR__, "output", task, name)
        mkpath(output_dir)
        _ = run_simulation_communication_perfect_vision(
            problem,
            output_dir,
            ess_thresh,
            num_particles,
            ground_truth_rewards,
            T
        )
    end

    task = "gpt4o"
    for problem in problems
        name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
        output_dir = joinpath(@__DIR__, "output", task, name)
        mkpath(output_dir)
        _ = run_simulation_gpt4o(
            problem,
            output_dir,
            ground_truth_rewards,
            T
        )
    end

end

main()