using Gen, GenGPT3
using Random
import Gen: ParticleFilterState
import GenParticleFilters: pf_initialize, pf_update!, pf_resample!, pf_rejuvenate!, get_norm_weights, get_traces, effective_sample_size

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(model="davinci-002", stop="\n", max_tokens=512, temperature=1)

global EXAMPLES_PICKUP = [
    ("Gem: red\nReward: +5", "I picked up a red gem and got +5 reward!"),
    ("Gem: blue\nReward: +2", "Blue gave me +2 reward."),
    ("Gem: yellow\nReward: +1", "yellow gem gave me +1 reward."),
    ("Gem: green\nReward: -1", "I got a -1 reward from the green gem."),
    ("Gem: red\nReward: +5", "Wow! A red gem just gave me a +5 reward boost!"),
    ("Gem: blue\nReward: +2", "Found a blue gem. Nice +2 reward."),
    ("Gem: yellow\nReward: +1", "Picked up a yellow, small +1 reward but still good."),
    ("Gem: green\nReward: -1", "Ouch, green gem with a -1 penalty."),
    ("Gem: red\nReward: +5", "Jackpot! red gem with a solid +5 reward."),
    ("Gem: blue\nReward: +2", "Not bad, blue gem adding +2 to my score."),
    ("Gem: yellow\nReward: +1", "yellow gem, just a +1 bump but I'll take it."),
    ("Gem: green\nReward: -1", "Darn, picked up a green. There goes 1 point."),
]
global EXAMPLES_NO_PICKUP = [
    ("Gem: none\nReward: +0", "I haven't come across a gem."),
    ("Gem: none\nReward: +0", "I haven't seen one yet."),
    ("Gem: none\nReward: +0", "No gems in sight so far."),
    ("Gem: none\nReward: +0", "Still searching for gems."),
    ("Gem: none\nReward: +0", "Haven't found any gems to pick up."),
    ("Gem: none\nReward: +0", "My inventory is empty, no gems collected."),
    ("Gem: none\nReward: +0", "I'm yet to encounter any gems on my path."),
    ("Gem: none\nReward: +0", "No luck finding gems at the moment."),
    ("Gem: none\nReward: +0", "I'm gem-less right now."),
    ("Gem: none\nReward: +0", "My gem count remains at zero."),
    ("Gem: none\nReward: +0", "The search for gems continues, none found yet."),
    ("Gem: none\nReward: +0", "I'm still on the lookout for gems."),
]

# Random.seed!(0)
shuffle!(EXAMPLES_PICKUP)
shuffle!(EXAMPLES_NO_PICKUP)

"""
    construct_prompt(context::String, examples::Vector{Tuple{String, String}})

    Construct a prompt for the GPT-3 model based on a context and examples.
"""
function construct_prompt(context::String, examples::Vector{Tuple{String, String}})
    example_strs = ["$ctx\nUtterance: $utt" for (ctx, utt) in examples]
    example_str = join(example_strs, "\n")
    prompt = "$example_str\n$context\nUtterance:"
    return prompt
end

@gen function agent_model(T::Int, num_agents::Int, possible_gems::Vector{Symbol}, possible_rewards::Vector{Int})
    # Reward for each gem type
    rewards::Dict{Symbol, Int} = Dict()
    for gem in possible_gems
        rewards[gem] = {:reward => Symbol(gem)} ~ labeled_uniform(possible_rewards)
    end

    for t = 1:T
        # Potential gem pickup and utterance by this agent
        gem_pickup = {t => :self => :gem_pickup} ~ bernoulli(0.5)
        if gem_pickup
            gem = {t => :self => :gem} ~ labeled_uniform(possible_gems)
            {t => :self} ~ utterance_model(gem, rewards[gem])
        end

        # Agent observes utteraces of the other agents (from the previous timestep)
        for i in 1:num_agents-1
            spoke = {t => :other_agents => i => :spoke} ~ bernoulli(0.5)
            if spoke
                gem = {t => :other_agents => i => :gem} ~ labeled_uniform(possible_gems)
                {t => :other_agents => i} ~ utterance_model(gem, rewards[gem])
            end
        end
    end

    return rewards
end

@gen function utterance_model(gem::Symbol, reward::Int)
    global EXAMPLES_PICKUP, EXAMPLES_NO_PICKUP
    context = "Gem: $gem\nReward: $reward"
    prompt = construct_prompt(context, EXAMPLES_PICKUP)
    utterance ~ gpt3(prompt)
    return utterance
end
