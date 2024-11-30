function update_beliefs_communication(pf_state, current_timestep, num_agents, possible_gems, possible_rewards, observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(agent_model_communication, (current_timestep, num_agents, possible_gems, possible_rewards), observation, num_particles)
    else
        # # Perform resampling if the effective sample size is below the threshold
        # if effective_sample_size(pf_state) < ess_thresh * num_particles
        #     println("Resampling")
        #     pf_resample!(pf_state, :residual)
        # end
        
        num_threads = Threads.nthreads()
        worker_indices = [((i-1)*num_particles÷num_threads + 1):(i*num_particles÷num_threads) for i in 1:num_threads]
        
        Threads.@threads for i in 1:length(worker_indices)
            indices = worker_indices[i]
            substate = view(pf_state, indices)
            pf_update!(substate, (current_timestep, num_agents, possible_gems, possible_rewards), (UnknownChange(),), observation)
        end
    end
    return pf_state
end

function update_beliefs_no_communication(pf_state, current_timestep, num_agents, possible_gems, possible_rewards, observation, num_particles, ess_thresh)
    if pf_state === nothing
        pf_state = pf_initialize(agent_model_no_communication, (current_timestep, num_agents, possible_gems, possible_rewards), observation, num_particles)
    else
        # if effective_sample_size(pf_state) < ess_thresh * num_particles
        #     println("Resampling")
        #     pf_resample!(pf_state, :residual)
            
        # end
        num_threads = Threads.nthreads()
        worker_indices = [((i - 1) * num_particles ÷ num_threads + 1):(i * num_particles ÷ num_threads) for i in 1:num_threads]
        
        Threads.@threads for i in 1:length(worker_indices)
            indices = worker_indices[i]
            substate = view(pf_state, indices)
            pf_update!(substate, (current_timestep, num_agents, possible_gems, possible_rewards), (), observation)
        end
    end
    return pf_state
end









function enum_inference(
    model::GenerativeFunction, model_args::Tuple,
    observations::ChoiceMap, latent_addrs, latent_values
)
    @assert length(latent_addrs) == length(latent_values)
    # Construct iterator over combinations of latent values
    latents_iter = Iterators.product(latent_values...)
    # Generate a trace for each possible combination of latent values
    traces = map(latents_iter) do latents
        constraints = choicemap()
        for (addr, val) in zip(latent_addrs, latents)
            constraints[addr] = val
        end
        constraints = merge(constraints, observations)
        tr, _ = Gen.generate(model, model_args, constraints)
        return tr
    end
    # Compute the log probability of each trace
    logprobs = map(Gen.get_score, traces)
    # Compute the log marginal likelihood of the observations
    lml = logsumexp(logprobs)
    # Compute the (marginal) posterior probabilities for each latent variable
    latent_logprobs = Dict(
        addr => ([logsumexp(lps) for lps in eachslice(logprobs, dims=i)] .- lml)
        for (i, addr) in enumerate(latent_addrs)
    )
    latent_probs = Dict(addr => exp.(lp) for (addr, lp) in latent_logprobs)
    return (
        traces = traces,
        logprobs = logprobs,
        latent_logprobs = latent_logprobs,
        latent_probs = latent_probs,
        latent_addrs = latent_addrs,
        lml = lml
    )
end



function enum_inference_step(
    prev_results::NamedTuple, new_model_args::Tuple, new_observations::ChoiceMap
)
    t = new_model_args[1]
    num_agents = new_model_args[2]
    possible_gems = new_model_args[3]
    possible_rewards = new_model_args[4]

    # Check if any agent spoke
    any_spoke = false
    speaking_agent = nothing
    for i in 1:num_agents-1
        if Gen.has_value(new_observations, (t => :other_agents => i => :spoke)) &&
            new_observations[t => :other_agents => i => :spoke]
            any_spoke = true
            speaking_agent = i
            break
        end
    end

    argdiffs = map(_ -> UnknownChange(), new_model_args)

    if !any_spoke
        # If no agent spoke, just update each trace
        traces = map(prev_results.traces) do prev_trace
            trace, _, _, _ = Gen.update(prev_trace, new_model_args, argdiffs, new_observations)
            return trace
        end
    else
        # If agent spoke, generate all possibilities
        traces = []
        for prev_trace in prev_results.traces
            for gem in possible_gems
                # Create a new choicemap instead of copying
                new_obs = new_observations
                new_obs[(t => :other_agents => speaking_agent => :gem)] = gem                
                trace, _, _, _ = Gen.update(prev_trace, new_model_args, argdiffs, new_obs)
                push!(traces, trace)
            end
        end
    end

    # Get logprobs for all traces
    logprobs = map(Gen.get_score, traces)
    lml = logsumexp(logprobs)

    # For each latent address (reward for each gem color)...
    latent_logprobs = Dict()
    latent_probs = Dict()
    
    for addr in prev_results.latent_addrs
        # Group traces by their reward value for this address
        reward_groups = Dict{Int, Vector{Float64}}()
        
        # For each trace and its logprob...
        for (trace, lp) in zip(traces, logprobs)
            reward = trace[addr]  # Get reward value for this gem color
            if !haskey(reward_groups, reward)
                reward_groups[reward] = Float64[]
            end
            push!(reward_groups[reward], lp)
        end
        
        # Calculate marginal probability for each possible reward value
        addr_logprobs = map(possible_rewards) do r
            if haskey(reward_groups, r)
                # Sum (in log space) over all traces with this reward value
                logsumexp(reward_groups[r]) - lml
            else
                -Inf  # If we never saw this reward value
            end
        end
        
        latent_logprobs[addr] = addr_logprobs
        latent_probs[addr] = exp.(addr_logprobs)
    end

    return (
        traces = traces,
        logprobs = logprobs,
        latent_logprobs = latent_logprobs,
        latent_probs = latent_probs,
        latent_addrs = prev_results.latent_addrs,
        lml = lml
    )
end