"""
To debug the inference algorithms, we set up a minimal possible experiment
"""

using MultiAgentCommunication
using DotEnv
using PDDL
using CSV
using DataFrames
using Gen

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
        observed_reward = Int[],
        problem_name = String[],
        repeat = Int[],
    )
    CSV.write(csv_filename, df)
end


function debug_simulation()

    overlay = DotEnv.config()
    api_key = get(overlay, "OPENAI_API_KEY", nothing)
    ENV["OPENAI_API_KEY"] = api_key

    ############# Parameters #############
    # TODO: input these parameters with an argsparser
    problem = joinpath(@__DIR__, "..", "src/problems/simple/2.pddl")
    output_dir = joinpath(@__DIR__, "output")
    ground_truth_rewards = Dict(:red => 1, :blue => -5, :yellow => 3, :green => 2)
    T = 100
    gridworld_only = false
    inference_type = "enum"
    ######################################

    mkpath(output_dir)
    csv_filename = joinpath(@__DIR__, output_dir, "results.csv")
    init_csv(csv_filename)

    results = run_simulation_communication_vision(
        problem,
        output_dir,
        ground_truth_rewards,
        T,
        gridworld_only,
        inference_type
    )

    # results = run_simulation_debug()

    CSV.write(csv_filename, results, append=true)

end

function debug_utterance_model()

    gem = :red
    reward = -5

    observation = Gen.choicemap()
    observation[:utterance => :output] = "Picked up a red, only +1 reward."

    tr, w = Gen.generate(utterance_model, (gem, reward), observation)

    println(display(Gen.get_choices(tr)))
    println(w)

end

debug_simulation()

# debug_utterance_model()
