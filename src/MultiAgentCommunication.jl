module MultiAgentCommunication

using PDDL, PlanningDomains, SymbolicPlanners
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using Random
using SymbolicPlanners: get_value
using DotEnv
include("utterances.jl")
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
    renderer = setup_renderer(domain, initial_state, agents, gridworld_only)
    canvas = renderer(domain, initial_state)

    # Output setup
    output_folder = joinpath(output_dir, problem_name)
    mkpath(output_folder)
    save(joinpath(output_folder, "initial_state.png"), canvas)

    # Main simulation loop
    actions = []
    t = 1
    combined_score = 0
    remaining_items = copy(items)
    possible_gems = collect(keys(ground_truth_rewards))
    possible_rewards = collect(values(ground_truth_rewards))
    num_gems_picked_up = Dict(agent => 0 for agent in agents)
    total_gems_picked_up = 0
    state = initial_state

    # Initialize beliefs and heuristics
    pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    gem_utilities = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)
    heuristics = [VisionGemHeuristic(agent, gem_utilities[agent]) for agent in agents]

    # Initialize planners
    planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]
    
    while !isempty(remaining_items) && t <= T
        for (i, agent) in enumerate(agents)
            goals = PDDL.Term[]
            rewards = Float64[]        
            current_beliefs = gem_utilities[agent]
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
            action = best_action(sol.value_policy, state, agent)
            state = transition(domain, state, action; check=true)

            if action.name == :pickup
                item = action.args[2].name
                println("Step $t:")
                println("       $agent picked up $item.")
                remaining_items = filter(x -> x != item, remaining_items)
                gem = parse_gem(String(item))
                num_gems_picked_up[agent] += 1
                total_gems_picked_up += 1
                reward = ground_truth_rewards[gem]
                combined_score += reward
                println("       $agent received $reward score.")
                println("       Combined score is now $combined_score.")

                observation = Gen.choicemap()
                observation[(1 => :gem_pickup)] = true
                observation[(1 => :gem)] = gem
                observation[(:reward => Symbol(gem))] = reward
                tr, _ = Gen.generate(utterance_model, (1, possible_gems, possible_rewards), observation)
                utterance = Gen.get_retval(tr)[1]
                println("       $agent communicated: $utterance.")
                
                alt_observation = Gen.choicemap()
                gem_count = num_gems_picked_up[agent]
                alt_observation[gem_count => :utterance => :output] = utterance
                alt_observation[gem_count => :gem_pickup] = true
                alt_observation[gem_count => :gem] = gem

                # Update beliefs
                pf_states[agent], current_pf_state = update_individual_beliefs(pf_states[agent], gem_count, possible_gems, possible_rewards, alt_observation, num_particles, ess_thresh)

                # Calculate and update gem utilities
                top_rewards = get_top_weighted_rewards(current_pf_state, 10, possible_gems)
                gem_certainty = quantify_gem_certainty(top_rewards)
                utilities, certainties = calculate_gem_utility(gem_certainty)

                gem_utilities[agent] = utilities

                print_estimated_rewards(agents, gem_utilities, certainties)
            end

            push!(actions, action)
        end
        t += 1
    end

    # Generate and save animation
    anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
    save(joinpath(output_folder, "plan.mp4"), anim)

    return combined_score
end

function setup_renderer(domain, initial_state, agents, gridworld_only)
    return PDDLViz.GridworldRenderer(
        resolution = (600,1100),
        has_agent = false,
        obj_renderers = Dict{Symbol, Function}(
            key => (d, s, o) -> begin
                if key == :agent
                    PDDLViz.MultiGraphic(
                        PDDLViz.RobotGraphic(color = :slategray),
                        PDDLViz.TextGraphic(
                            string(o.name)[end:end], 0.3, 0.2, 0.5,
                            color = :black, font = :bold
                        )
                    )
                else
                    PDDLViz.GemGraphic(color = key)
                end
            end
            for key in [:agent, :red, :yellow, :blue, :green]
        ),
        show_inventory = !gridworld_only,
        inventory_fns = [
            (d, s, o) -> s[PDDL.Compound(:has, [PDDL.Const(agent), o])] for agent in agents
        ],
        inventory_types = [:item for agent in agents],
        inventory_labels = ["$agent Inventory" for agent in agents],
        show_vision = !gridworld_only,
        vision_fns = [
            (d, s, o) -> s[PDDL.Compound(:visible, [PDDL.Const(agent), o])] for agent in agents
        ],
        vision_types = [:item for agent in agents],
        vision_labels = ["$agent Vision" for agent in agents],
    )
end

function update_shared_beliefs(pf_states, gem_count, possible_gems, possible_rewards, alt_observation, num_particles, ess_thresh, total_gems_picked_up)
    if pf_states === nothing
        pf_states = pf_initialize(utterance_model, (gem_count, possible_gems, possible_rewards), alt_observation, num_particles)
    else
        if effective_sample_size(pf_states) < ess_thresh * num_particles
            pf_resample!(pf_states, :stratified)
            rejuv_sel = select()
            pf_rejuvenate!(pf_states, mh, (rejuv_sel,))
        end
        pf_update!(pf_states, (total_gems_picked_up, possible_gems, possible_rewards), (UnknownChange(),), alt_observation)
    end
    return pf_states, pf_states
end

function update_individual_beliefs(pf_state, gem_count, possible_gems, possible_rewards, alt_observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(utterance_model, (gem_count, possible_gems, possible_rewards), alt_observation, num_particles)
    else
        if effective_sample_size(pf_state) < ess_thresh * num_particles
            pf_resample!(pf_state, :stratified)
            rejuv_sel = select()
            pf_rejuvenate!(pf_state, mh, (rejuv_sel,))
        end
        pf_update!(pf_state, (gem_count, possible_gems, possible_rewards), (UnknownChange(),), alt_observation)
    end
    return pf_state, pf_state
end

function print_estimated_rewards(agents, gem_utilities, certainties)
    for agent in agents
        println("       $agent's Estimated Rewards:")
        for (gem, value) in gem_utilities[agent]
            println("              $gem: value = $value, certainty = $(round(certainties[gem], digits=2))")
        end
    end
end
    
end # module