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
    # Update previous traces with the new arguments and observations
    argdiffs = map(_ -> UnknownChange(), new_model_args)
    traces = map(prev_results.traces) do prev_trace
        trace, _, _, _ =
            Gen.update(prev_trace, new_model_args, argdiffs, new_observations)
        return trace
    end
    # Compute the log probability of each trace
    logprobs = map(Gen.get_score, traces)
    # Compute the log marginal likelihood of the observations
    lml = logsumexp(logprobs)
    # Compute the (marginal) posterior probabilities for each latent variable
    latent_logprobs = Dict(
        addr => ([logsumexp(lps) for lps in eachslice(logprobs, dims=i)] .- lml)
        for (i, addr) in enumerate(prev_results.latent_addrs)
    )
    latent_probs = Dict(addr => exp.(lp) for (addr, lp) in latent_logprobs)
    return (
        traces = traces,
        logprobs = logprobs,
        latent_logprobs = latent_logprobs,
        latent_probs = latent_probs,
        latent_addrs = prev_results.latent_addrs,
        lml = lml
    )
end
