# using PyCall        # loading in PyCall and PythonOperations.py first; need to do this b/c of dependency issues;
# @pyinclude(joinpath("model-backend", "PythonOperations.py"))
# PYTHON_OPS = py"PythonOps"()
# note dependency problem with pycall and geodataframes below. proj.db issues.

# loading in other packages now
using Agents
using DataFrames
using GeoDataFrames
using ArchGDAL
using ProgressMeter
using CSV
using BSON
using ArgParse
using Random
using Distributions
using Flux
using StaticArrays
using DeepQLearning
using POMDPPolicies
using StatsBase
import GeoFormatTypes as GFT
using JSON
using DataStructures: OrderedDict
using EllipsisNotation

include(joinpath("InputStructs.jl"))
# include(joinpath("model-backend", "DQN.jl"))
include(joinpath("model-backend", "MyModel.jl"))
include(joinpath("model-backend", "ResidentialAgent.jl"))
include(joinpath("model-backend", "MiscFunc.jl"))
include(joinpath("model-backend", "MyParcelSpace.jl"))
include(joinpath("model-backend", "weights.jl"))
# include(joinpath("train_agents.jl"))


"""
TODOs:
    - implement HUA
"""
function main()
    input_struct = InputStruct()
    parsed_args = parse_commandline()
    (parsed_args["model_runname"]!=nothing) && (input_struct.model_runname = parsed_args["model_runname"])
    println("\nRunning Model: $(input_struct.model_runname)")

    path_to_input_dir = joinpath(pwd(), "model-runs", input_struct.model_runname, "input")
    run_model(input_struct, parsed_args, path_to_input_dir)
end

function run_model(input_struct, parsed_args, path_to_input_dir)
    println("starting model run")
    
    (parsed_args["end_sim"] != nothing) && (input_struct.n_iterations = parsed_args["end_sim"])
    # --- Running model
    p = ProgressBar(input_struct.n_iterations; color=:black)
    for i = parsed_args["start_sim"]:input_struct.n_iterations
        input_struct.agent_family_weights = InputStruct().agent_family_weights  # resetting input struct weights; if not, then these are kept as the last iterations' values. 
        model = initialize(path_to_input_dir, input_struct, parsed_args, seed=input_struct.seed, iter=i)    # initializing model
        data_a, data_m, data_s = my_run!(
                                    model,
                                    input_struct.n_years;
                                    adata=get_agent_save_data(),
                                    mdata=get_model_save_data(),
                                    sdata=get_space_save_data(),
                                    )
        # --- saving results for iteration
        # fn_agnts = "df_agnts_$(i)_sc$(input_struct.slr_scenario)_ne$(input_struct.slr_ne).csv"
        # write_out(data_a, model, fn_agnts)

        fn_model = "df_model_$(i)_sc$(input_struct.slr_scenario)_ne$(input_struct.slr_ne).csv"
        write_out(data_m, model, fn_model)

        # fn_space = "df_space_$(i)_sc$(input_struct.slr_scenario)_ne$(input_struct.slr_ne).csv"
        # write_out(data_s, model, fn_space)

        close_model!(model, input_struct.model_runname)
        next!(p)
    end
end


"""
    my_run!()
Custom model run function. 
Started from example on Agents.jl docs
Needed to customize to collect space (pacel) data during model time steps
"""
function my_run!(model,
                n;
                when = true,
                when_model = when,
                mdata = nothing,
                adata = nothing,
                sdata = nothing,
                )

    df_agent = init_agent_dataframe(model, adata)
    df_model = init_model_dataframe(model, mdata)
    df_space = init_space_dataframe(model, sdata)
    s = 0    
    while Agents.until(s, n, model)
          collect_agent_data!(df_agent, model, adata)
          collect_model_data!(df_model, model, mdata)
          collect_space_data!(df_space, model, sdata, s)
      step!(model, 1)
      s += 1
    end
    return df_agent, df_model, df_space
end


"""
    init_space_dataframe()
Function to initialize space dataframe.
Used to store space (parcel) results for output
"""
function init_space_dataframe(model::ABM, properties::AbstractArray)
    std_headers = 2

    headers = Vector{String}(undef, std_headers + length(properties))
    headers[1] = "step"
    headers[2] = "guid"

    for i in 1:length(properties)
        headers[i+std_headers] = dataname(properties[i])
    end
    types = Vector{Vector}(undef, std_headers + length(properties))
    types[1] = Int[]
    types[2] = String[]
    for (i, field) in enumerate(properties)
        types[i+2] = eltype(model.prcl_df[!,field])[]
    end

    push!(properties, :guid)
    push!(properties, :step)
    return DataFrame(types, headers)
end


"""
    collect_space_data!()
Function to collect space (parcel) data from model at each time step
"""
function collect_space_data!(df, model, properties::Vector, step::Int = 0)
    dd = model.prcl_df
    dd[!,:step] = fill(step, size(dd,1))
    dd = dd[!,properties]
    append!(df, dd)
    return df
end


"""
    parse_commandline()
Function to pass option values into the model from the command line
Created to start simulation from user-specified iteration
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--start_sim"
            help = "option to start simulation at iteration other than 1"
            default = 1
            arg_type = Int
            required = false
        "--end_sim"
            help = "option to end simulation at iteration other than 1"
            default = nothing
            arg_type = Int
            required = false
        "--model_runname"
            help = "option to run specific model runname"
            default = nothing
            arg_type = String
            required = false
        "--train"
            help = "option to run under training conditions or not"
            default = nothing
            arg_type = Bool
            required = false
        "--sub_runname"
            help = "which sub_runname to run"
            default = nothing
            arg_type = String
            required = false
        "--output_dir"
            help = "where to write output"
            default = nothing
            arg_type = String
            required = false
        "--training_dir"
            help = "where to write training results"
            default = nothing
            arg_type = String
            required = false
    end
    return parse_args(s)
end


# ------------
main()



