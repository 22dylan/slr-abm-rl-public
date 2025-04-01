# file for storing inputs used in code; input structs for both model and DQN


Base.@kwdef mutable struct InputStruct
    model_runname::String="status-quo"            # model runname
    sub_runname::String="scInt-ne0p5"              # model subruname; used in testing agents

    train::Bool=false
    n_years::Int64=76                       # number of years per iteration/episode
    n_iterations::Int64=10                 # number of iterations (or episodes) in model runs
    slr_scenario::String="Int"              # noaa sea level rise (slr) scenario (Low, IntLow, Int, IntHigh, High) or "train"
    slr_ne::String="0.5"                    # noaa sea level rise (slr) non-exceedance prob (ne) (0.17, 0.5, 0.83) or "train"
    seed::Int64=1337                        # seed for simulations
    model_start_year::Int64=2025            # model start year; used in slr exposure
    agent_view_radius::Int64=5              # agent's view radius

    # agent_families::Array{Symbol}=[:agent1, :agent2, :agent4, :agent5, :agent7]
    agent_families::Array{Symbol}=[:agent1, :agent2, :agent3, :agent4, :agent5, :agent6, :agent7, :agent8]
    agent_family_weights::Array{Float64}=Float64[]
    agent_network_episode::String="1000000"
end


Base.@kwdef mutable struct DQNParam
    ## values for DQN
    t::Int64=0                          # counter for total number of training steps taken
    discount::Float32=0.98              # discount factor (0->agent myopic), (1->agent values later rewards)
    learning_rate::Float64=0.0001       # learning rate; alpha (default=1e-4)
    target_update_freq::Int64=1_000     # frequency at which target network is updated (default=500)
    batch_size::Int64=64                # batch size sample from replay buffer (default=32)
    train_freq::Int64=4                 # frequency at which active network is updated (default=4)
    buffer_size::Int64=10_000           # size of experience replay buffer (default=1000)
    train_start::Int64=5_000            # number of steps used to fill in replay buffer initially (default=200)

    ## the following are for logging training results
    n_episodes_eval::Int64=1_000        # number of states to evaluate network
    eval_freq::Int64=10_000             # frequency (episodes) at which to evaluate the network (default=100)
    save_ntwk_freq::Int64=100_000       # frequency (episodes) at which to save neural network (deault=1000)
    write_freq::Int64=10_000            # frequency (episodes) at which to write the train log

    # for linear epsislon decay
    eps_max::Float32=1.0                # epsilon greedy maximum value
    eps_min::Float32=0.1                # epsilon greedy minimum value
    eps::Float32=1.0                    # initializing epsilon
    n_steps_decay::Int64=100_000        # number of steps (or episodes) to decay epsilon to minimum
    
    # for double Q and prioritized replay
    double_q::Bool=true                 # double q learning update (default=true)
    prioritized_replay::Bool=true       # enable prioritized experience replay (default=true)

    ## drs added values used during training    
    episode_log::Vector{Int64}=[0]
    train_steps::Vector{Int64}=[0]
    epsilon_log::Vector{Float64}=[0.0]
    buffer_size_log::Vector{Int64}=[0]
    episode_reward_log::Vector{Float64}=[0.0]
    eval_log::Vector{Float64}=[0.0]
    loss_log::Vector{Float64}=[0.0]
end
