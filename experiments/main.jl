using MultiAgentCommunication
using DotEnv
using PDDL
using CSV
using DataFrames

PDDL.Arrays.register!()

function init_csv(csv_filename)
    mkpath(dirname(csv_filename))
    isfile(csv_filename) && rm(csv_filename)
    df = DataFrame(
        timestep = Int[],
        agent = Int[],
        score = Float64[],
        gems_picked_up = Int[],
        red_m5 = Float64[], red_1 = Float64[], red_2 = Float64[], red_3 = Float64[],
        blue_m5 = Float64[], blue_1 = Float64[], blue_2 = Float64[], blue_3 = Float64[],
        green_m5 = Float64[], green_1 = Float64[], green_2 = Float64[], green_3 = Float64[],
        yellow_m5 = Float64[], yellow_1 = Float64[], yellow_2 = Float64[], yellow_3 = Float64[],
        pickup = String[],
        utterance = String[],
        problem_name = String[],
        repeat = Int[]
    )
    CSV.write(csv_filename, df)
end

function main()

    overlay = DotEnv.config()
    api_key = get(overlay, "OPENAI_API_KEY", nothing)
    ENV["OPENAI_API_KEY"] = api_key

    problem_folder = joinpath(@__DIR__, "..", "src/problems/medium")
    files = readdir(problem_folder)
    problems = [joinpath(problem_folder, file) for file in files]
    

    ############# Parameters #############
    ess_thresh = 0.3
    num_particles = 100
    ground_truth_rewards = Dict(:red => 1, :blue => -5, :yellow => 3, :green => 2)
    T = 50
    repeats = 1
    ######################################


    # Run communication_vision
    task = "communication_vision"
    csv_filename = joinpath(@__DIR__, "output", task, "results.csv")
    init_csv(csv_filename)
    for problem in problems
        for repeat in 1:repeats
            name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]     
            output_dir = joinpath(@__DIR__, "output", task, name)
            mkpath(output_dir)
            results = run_simulation_communication_vision(
                problem,
                output_dir,
                ess_thresh,
                num_particles,
                ground_truth_rewards,
                T
            )
            results.problem_name .= name
            results.repeat .= repeat
            CSV.write(csv_filename, results, append=true)
        end
    end

    # Run no_communication_vision
    task = "no_communication_vision"
    csv_filename = joinpath(@__DIR__, "output", task, "results.csv")
    init_csv(csv_filename)
    for problem in problems
        for repeat in 1:repeats
            name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
            output_dir = joinpath(@__DIR__, "output", task, name)
            mkpath(output_dir)
            results = run_simulation_no_communication_vision(
                problem,
                output_dir,
                ess_thresh,
                num_particles,
                ground_truth_rewards,
                T
            )
            results.problem_name .= name
            results.repeat .= repeat
            CSV.write(csv_filename, results, append=true)
        end
    end

    # Run communication_restricted_vision
    task = "communication_restricted_vision"
    csv_filename = joinpath(@__DIR__, "output", task, "results.csv")
    init_csv(csv_filename)
    for problem in problems
        for repeat in 1:repeats
            name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
            output_dir = joinpath(@__DIR__, "output", task, name)
            mkpath(output_dir)
            results = run_simulation_communication_restricted_vision(
                problem,
                output_dir,
                ess_thresh,
                num_particles,
                ground_truth_rewards,
                T
            )
            results.problem_name .= name
            results.repeat .= repeat
            CSV.write(csv_filename, results, append=true)
        end
    end

    # Run communication_perfect_vision
    task = "communication_perfect_vision"
    csv_filename = joinpath(@__DIR__, "output", task, "results.csv")
    init_csv(csv_filename)
    for problem in problems
        for repeat in 1:repeats
            name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
            output_dir = joinpath(@__DIR__, "output", task, name)
            mkpath(output_dir)
            results = run_simulation_communication_perfect_vision(
                problem,
                output_dir,
                ess_thresh,
                num_particles,
                ground_truth_rewards,
                T
            )
            results.problem_name .= name
            results.repeat .= repeat
            CSV.write(csv_filename, results, append=true)
        end
    end

    # Run gpt4o
    task = "gpt4o"
    csv_filename = joinpath(@__DIR__, "output", task, "results.csv")
    init_csv(csv_filename)
    for problem in problems
        for repeat in 1:repeats
            name = split(problem, "/")[end-1] * "/" * split(split(problem, "/")[end], ".")[1]
            output_dir = joinpath(@__DIR__, "output", task, name)
            mkpath(output_dir)
            results = run_simulation_gpt4o(
                problem,
                output_dir,
                ground_truth_rewards,
                T
            )
            results.problem_name .= name
            results.repeat .= repeat
            CSV.write(csv_filename, results, append=true)
        end
    end

end

main()
