# file for DQN training
struct MyDQExperience{N <: Real,T <: Real, A<:AbstractArray}
    s::A
    a::N
    r::T
    sp::A
    done::Bool
end


mutable struct MyPrioritizedReplayBuffer{N<:Integer, T<:AbstractFloat,CI,Q,A<:AbstractArray{T}}
    max_size::Int64
    batch_size::Int64
    rng::AbstractRNG
    α::Float32
    β::Float32
    ϵ::Float32
    _curr_size::Int64
    _idx::Int64
    _priorities::Vector{T}
    _experience::Vector{MyDQExperience{N,T,Q}}

    _s_batch::A
    _a_batch::Vector{CI}
    _r_batch::Vector{T}
    _sp_batch::A
    _done_batch::Vector{T}
    _weights_batch::Vector{T}
end


function MyPrioritizedReplayBuffer(agent_type::Type{T},
                                max_size::Int64,
                                batch_size::Int64;
                                rng::AbstractRNG = MersenneTwister(0),
                                α::Float32 = 6f-1,
                                β::Float32 = 4f-1,
                                ϵ::Float32 = 1f-3) where T<:AbstractAgent
    o = define_state(agent_type)
    s_dim = size(o)
    experience = Vector{MyDQExperience{Int32, Float32, typeof(o)}}(undef, max_size)
    priorities = Vector{Float32}(undef, max_size)
    _s_batch = zeros(Float32, s_dim..., batch_size)
    _a_batch = [CartesianIndex(0,0) for i=1:batch_size]
    _r_batch = zeros(Float32, batch_size)
    _sp_batch = zeros(Float32, s_dim..., batch_size)
    _done_batch = zeros(Float32, batch_size)
    _weights_batch = zeros(Float32, batch_size)
    return MyPrioritizedReplayBuffer(max_size, batch_size, rng, α, β, ϵ, 0, 1, priorities, experience,
                _s_batch, _a_batch, _r_batch, _sp_batch, _done_batch, _weights_batch)
end



function read_dqn_discount!(dqn_param::DQNParam, input_struct::InputStruct)
    f = joinpath("agent-weights.csv")
    df = read_csv(f)
    df_ = df[findfirst(df.runname.==input_struct.sub_runname), :]

    dqn_param.discount = df_[:discount]
end
"""
    epsilon_decay(step, eps_max, eps_min, n_steps_decay)
linear epsilon decay function
"""
function epsilon_decay(step, eps_max, eps_min, n_steps_decay)
    m = (eps_max - eps_min) / (-n_steps_decay)
    y = (m*step)+eps_max
    (y<eps_min) && (y = eps_min)
    return y
end


"""
    add_exp!()
add experience to replay buffer; taken from DeepQLearning.jl
"""
function add_exp!(r::MyPrioritizedReplayBuffer, expe::MyDQExperience, td_err::T=abs(expe.r)) where T
    @assert td_err + r.ϵ > 0.
    priority = (td_err + r.ϵ)^r.α
    r._experience[r._idx] = expe
    r._priorities[r._idx] = priority
    r._idx = mod1((r._idx + 1),r.max_size)
    if r._curr_size < r.max_size
        r._curr_size += 1
    end
end

# add_dim(x::MArray) = reshape(x, (size(x)...,1))

"""
    decide_action(agent::AbstractAgent, model::ABM, train::Bool)
returns an action for the agent, if train is passed in
"""
function decide_action!(agent::T, model::ABM, agent_dict::Dict; train::Bool) where T<:AbstractAgent
    (train==false) && (return decide_action!(agent, agent_dict[agent.family][:network]))
    if rand(abmrng(model)) < agent_dict[agent.family][:DQNParam].eps
        agent.action = wsample(abmrng(model), agent.actions, agent.action_weights)    # weighted sample of agent's actions; weighted to keep agent in parcel longer
        return
    else
        active_q = agent_dict[agent.family][Symbol("ntwk_$(agent.ntwk_num)")][:active_q]
        decide_action!(agent, active_q)  # best action using active_q network
        return
    end
end

function decide_action_train!(agent::T, rng::MersenneTwister, dqn_param::DQNParam, active_q::Chain) where T<:AbstractAgent
    if rand(rng) < dqn_param.eps
        agent.action = wsample(rng, agent.actions, agent.action_weights)    # weighted sample of agent's actions; weighted to keep agent in parcel longer
        return
    else
        decide_action!(agent, active_q)  # best action using active_q network
        return
    end
end



"""
    dqn_train_step()
deep q-network training step; taken and adapted from DeepQLearning.jl
"""
function dqn_train_step!(agent::T, model::ABM, agent_dict::Dict, exp::MyDQExperience) where T<:AbstractAgent
    replay   = agent_dict[agent.family][:replay]
    DQNParam = agent_dict[agent.family][:DQNParam]
    active_q = agent_dict[agent.family][Symbol("ntwk_$(agent.ntwk_num)")][:active_q]
    target_q = agent_dict[agent.family][Symbol("ntwk_$(agent.ntwk_num)")][:target_q]

    training_df = agent_dict[agent.family][:training_df]
    train_steps = training_df[training_df[!,:ntwk_num].==agent.ntwk_num, :train_steps][1]
    optimizer = Adam(DQNParam.learning_rate)

    add_exp = check_add_experience_to_replay(exp, model)

    if DQNParam.prioritized_replay
        (add_exp) && add_exp!(replay, exp, abs(exp.r))
    else
        (add_exp) && add_exp!(replay, exp, 0f0)
    end

    DQNParam.episode_reward_log[end] += exp.r
    if (train_steps%DQNParam.train_freq == 0) && (replay._curr_size > replay.batch_size) # todo: drs added this; only start training after replay buffer is larger than batchsize
        hs = hiddenstates(active_q)
        loss_val, grad_val = batch_train!(model, optimizer, active_q, target_q, replay, DQNParam)
        sethiddenstates!(active_q, hs)
        DQNParam.loss_log[end] += loss_val
    end
    if train_steps%DQNParam.target_update_freq == 0
        weights = Flux.params(active_q)
        Flux.loadparams!(target_q, weights)
    end
    training_df[training_df[!,:ntwk_num].==agent.ntwk_num, :train_steps] .+= 1
    DQNParam.train_steps[end] += 1

    agent_dict[agent.family][:replay] = replay
    agent_dict[agent.family][:DQNParam] = DQNParam
    agent_dict[agent.family][Symbol("ntwk_$(agent.ntwk_num)")][:active_q] = active_q
    agent_dict[agent.family][Symbol("ntwk_$(agent.ntwk_num)")][:target_q] = target_q
    agent_dict[agent.family][:training_df] = training_df

end

""" 
    hiddenstates(m)
returns the hidden states of all the recurrent layers of a model
taken from DeepQLearning.jl
""" 
function hiddenstates(m)
    return [l.state for l in m if l isa Flux.Recur]
end

"""
    batch_train!()
batch training network; taken from DeepQLearning.jl
"""
function batch_train!(model::ABM,
                      optimizer,
                      active_q::Chain, 
                      target_q::Chain,
                      replay::MyPrioritizedReplayBuffer,
                      DQNParam::DQNParam;
                     )

    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, importance_weights = sample(replay)
    p = Flux.params(active_q)

    loss_val = nothing
    td_vals = nothing

    γ = DQNParam.discount
    if DQNParam.double_q
        qp_values = active_q(sp_batch)
        target_q_values = target_q(sp_batch)
        best_a = [CartesianIndex(argmax(qp_values[..,i]), i) for i=1:DQNParam.batch_size]
        # best_a = [CartesianIndex(argmax(qp_values[fill(:,ndims(qp_values)-1)...,i]), i) for i=1:DQNParam.batch_size]
        q_sp_max = target_q_values[best_a]
    else
        q_sp_max = dropdims(maximum(target_q(sp_batch), dims=1), dims=1)
    end
    q_targets = r_batch .+ (1f0 .- done_batch) .* γ .* q_sp_max


    # for future gpu training: slight bottle neck here, ~0.001 sec; 961 allocations
    gs = Flux.gradient(p) do
        q_values = active_q(s_batch)        
        q_sa = q_values[a_batch]        
        td_vals = q_sa .- q_targets        
        loss_val = sum(huber_loss, importance_weights.*td_vals)        
        loss_val /= DQNParam.batch_size
    end

    grad_norm = globalnorm(p, gs)
    Flux.Optimise.update!(optimizer, p, gs)

    if DQNParam.prioritized_replay
        update_priorities!(replay, indices, td_vals)
    end
    return loss_val, grad_norm
end

"""
    StatsBase.sample(r::MyPrioritizedReplayBuffer)
sample from the replay buffer.
taken from DeepQLearning.jl
"""
function StatsBase.sample(r::MyPrioritizedReplayBuffer)
    @assert r._curr_size >= r.batch_size  
    @assert r.max_size >= r.batch_size      # could be checked during construction
    sample_indices = sample(r.rng, 1:r._curr_size, Weights(r._priorities[1:r._curr_size]), r.batch_size, replace=false)
    return get_batch(r, sample_indices)
end

"""
    get_batch(r::MyPrioritizedReplayBuffer, sample_indices)
returns sample from the replay buffer.
taken from DeepQLearning.jl
"""
# function get_batch(r::MyPrioritizedReplayBuffer, sample_indices::Vector{Int64})
#     @assert length(sample_indices) == size(r._s_batch)[end]
#     for (i, idx) in enumerate(sample_indices)
#         @inbounds begin
#             # r._s_batch[fill(:,ndims(r._s_batch)-1)...,1] = vec(r._experience[idx].s)    # note: drs added this; dimension issue
#             r._s_batch[:, i] = vec(r._experience[idx].s)
#             r._a_batch[i] = CartesianIndex(r._experience[idx].a, i)
#             r._r_batch[i] = r._experience[idx].r
#             # r._sp_batch[fill(:,ndims(r._sp_batch)-1)...,1] = vec(r._experience[idx].sp) # note: drs added this; dimension issue
#             r._sp_batch[:, i] = vec(r._experience[idx].sp)
#             r._done_batch[i] = r._experience[idx].done
#             r._weights_batch[i] = r._priorities[idx]
#         end
#     end
#     p = r._weights_batch ./ sum(r._priorities[1:r._curr_size])
#     weights = (r._curr_size * p).^(-r.β)
#     return r._s_batch, r._a_batch, r._r_batch, r._sp_batch, r._done_batch, sample_indices, weights
# end

function get_batch(r::MyPrioritizedReplayBuffer, sample_indices::Vector{Int64})
    @assert length(sample_indices) == size(r._s_batch)[end]
    for (i, idx) in enumerate(sample_indices)
        @inbounds begin
            r._s_batch[.., i] = vec(r._experience[idx].s)
            r._a_batch[i] = CartesianIndex(r._experience[idx].a, i)
            r._r_batch[i] = r._experience[idx].r
            r._sp_batch[.., i] = vec(r._experience[idx].sp)
            r._done_batch[i] = r._experience[idx].done
            r._weights_batch[i] = r._priorities[idx]
        end
    end
    p = r._weights_batch ./ sum(r._priorities[1:r._curr_size])
    weights = (r._curr_size * p).^(-r.β)
    return r._s_batch, r._a_batch, r._r_batch, r._sp_batch, r._done_batch, sample_indices, weights
end

"""
    globalnorm(p::Params, gs::Flux.Zygote.Grads)
returns the maximum absolute values in the gradients of W.
taken from DeepQLearning.jl
"""
function globalnorm(ps::Flux.Params, gs::Flux.Zygote.Grads)
    gnorm = 0f0
    for p in ps 
        gs[p] === nothing && continue 
        curr_norm = maximum(abs.(gs[p]))
        gnorm =  curr_norm > gnorm  ? curr_norm : gnorm
    end 
    return gnorm
end

"""
    update_priorities()
updates priorities in replay buffer
taken from DeepQLearning.jl
"""
function update_priorities!(r::MyPrioritizedReplayBuffer, indices::Vector{Int64}, td_errors::V) where V <: AbstractArray
    new_priorities = (abs.(td_errors) .+ r.ϵ).^r.α
    @assert all(new_priorities .> 0f0)
    r._priorities[indices] = new_priorities
end

"""
    sethiddenstates!(m, hs)
Given a list of hiddenstate, set the hidden state of each recurrent layer of the model m 
to what is in the list. 
The order of the list should match the order of the recurrent layers in the model.
taken from DeepQLearning.jl
"""
function sethiddenstates!(m, hs)
    i = 1
    for l in m
        if isa(l, Flux.Recur) 
            l.state = hs[i]
            i += 1
        end
    end
end
