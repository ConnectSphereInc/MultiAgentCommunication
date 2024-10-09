module MultiAgentCommunication

using PDDL, PlanningDomains, SymbolicPlanners
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using Random
using SymbolicPlanners: get_value
using DotEnv
include("agent.jl")
include("utils.jl")
include("heuristics.jl")

export run_simulation

function run_simulation(
    problem_name::String,
    ess_thresh::Float64,
    num_particles::Int,
    ground_truth_rewards::Dict{Symbol, Int},
    T::Int,
    output_dir::String,
    gridworld_only::Bool = false
)
    PDDL.Arrays.register!()

    # Load domain and problem
    domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
    problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name * ".pddl"))
    initial_state = initstate(domain, problem)
    
    # Retrieve objects before compilation
    items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
    agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]
    
    # Renderer setup
    renderer = setup_renderer(agents, gridworld_only)
    canvas = renderer(domain, initial_state)

    # Output setup
    output_folder = joinpath(output_dir, problem_name)
    mkpath(output_folder)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    combined_score = 0
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    total_gems_picked_up = 0
    state = initial_state

    # Initialize beliefs and heuristics optimistically
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    beliefs = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)
    heuristics = [VisionGemHeuristic(agent, beliefs[agent]) for agent in agents]

    # Initialize planners
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]
    
    # Initialize observations for each agent
    observations = Dict(agent => Gen.choicemap() for agent in agents)

    # Initialize previous utterances
    previous_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)

    # Main simulation loop
    t = 1
    while !isempty(remaining_items) && t <= T
        println("Step $t:")
        current_utterances = Dict{Symbol, Union{Nothing, String}}(agent => nothing for agent in agents)
        for (i, agent) in enumerate(agents)
            goals = PDDL.Term[]
            rewards = Float64[]        
            current_beliefs = beliefs[agent]
            for gem in remaining_items
                gem_obj = PDDL.Const(gem)
                color = Symbol(split(string(gem), "_")[1])
                reward = current_beliefs[color]
                heuristics[i].rewards[color] = reward

                if reward >= 0
                    push!(goals, PDDL.pddl"(has $agent $gem_obj)")
                    push!(rewards, reward)
                end
            end

            spec = MultiGoalReward(goals, rewards, 0.95)
            sol = planners[i](domain, state, spec)
            action = boltzmann_action(sol.value_policy, state, agent, 0.)
            state = transition(domain, state, action; check=true)

            # Clear previous observations and create new observation for this timestep
            observations[agent] = Gen.choicemap()
            observations[agent][(t => :self => :gem_pickup)] = false

            if action.name == :pickup
                item = action.args[2].name
                println("       $agent picked up $item.")
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                reward = ground_truth_rewards[gem]
                combined_score += reward
                println("       $agent received $reward score.")
                println("       Combined score is now $combined_score.")

                # Update the observations for pickup
                observations[agent][(t => :self => :gem_pickup)] = true
                observations[agent][(t => :self => :gem)] = gem
                # observations[agent][(t => :self => :reward_received)] = reward

                # Generate utterance
                utterance_tr, _ = Gen.generate(utterance_model, (gem, reward), Gen.choicemap())
                utterance = Gen.get_retval(utterance_tr)
                current_utterances[agent] = utterance
                observations[agent][(t => :self => :utterance => :output)] = utterance
                println("       $agent communicated: $utterance")
            end

            # Add other agents' utterances from the previous timestep to the observations
            other_agent_index = 1
            for other_agent in agents
                if other_agent != agent
                    if previous_utterances[other_agent] !== nothing
                        observations[agent][(t => :other_agents => other_agent_index => :spoke)] = true
                        observations[agent][(t => :other_agents => other_agent_index => :utterance => :output)] = previous_utterances[other_agent]
                    else
                        observations[agent][(t => :other_agents => other_agent_index => :spoke)] = false
                    end
                    other_agent_index += 1
                end
            end

            # Update beliefs for the current agent using only the most recent observation
            pf_states[agent] = update_beliefs(pf_states[agent], t, length(agents), possible_gems, possible_rewards, observations[agent], num_particles, ess_thresh)

            # Calculate and update gem utilities for the current agent
            current_pf_state = pf_states[agent]
            top_rewards = get_top_weighted_rewards(current_pf_state, 10, possible_gems)
            gem_certainty = quantify_gem_certainty(top_rewards)
            utilities, certainties = calculate_gem_utility(gem_certainty)
            beliefs[agent] = utilities

            print_estimated_rewards(agent, beliefs[agent], certainties)

            push!(actions, action)
        end

        # Update previous_utterances for the next timestep
        previous_utterances = current_utterances

        t += 1
    end

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    return combined_score
end

function update_beliefs(pf_state, current_timestep, num_agents, possible_gems, possible_rewards, observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(agent_model, (current_timestep, num_agents, possible_gems, possible_rewards), observation, num_particles)
    else
        # todo: fix resample and rejuvination
        # if effective_sample_size(pf_state) < ess_thresh * num_particles
        #     pf_resample!(pf_state, :residual)
        #     rejuv_sel = Gen.select()
        #     pf_rejuvenate!(pf_state, mh, (rejuv_sel,))
        # end
        pf_update!(pf_state, (current_timestep, num_agents, possible_gems, possible_rewards), (UnknownChange(),), observation)
    end
    return pf_state
end

end