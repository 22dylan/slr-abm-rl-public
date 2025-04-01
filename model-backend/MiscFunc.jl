# file for miscellaneous functions


binindices(edges, data) = searchsortedlast.(Ref(edges), data)

"""
    reading input folder; 
"""
function read_dir_CSV(path_to_input_dir::String)
    all_in_dir = readdir(path_to_input_dir)
    files = String[]    # files
    # dirs = String[]   # directories
    for f in all_in_dir
        if '.' in f
            if occursin("DS_Store", f) # ignoring DS_Store file
                continue
            end
        push!(files, f)
        # else
        #   push!(dirs, f)
        end
    end

    input_dict = Dict()
    for f in files
        key = split(f, ".")[1]
        f = joinpath(path_to_input_dir, f)
        if occursin("csv", f)
            input_dict[key] = read_csv(f)
        end
    end


    return input_dict
end


"""
    read_shpfile(path_to_input_dir, subdir)
Reads shapefile in the provided directory and subdirectory
path_to_input_dir: directory containing all input files and shapefiles
subdir: subdirectory within path_to_input_dir that needs to be read. Contains
    the actual shapefiles; note that the actual shapefile name isn't necessary,
    just the subdirectory that it exists in
Directory structure as follows:
    path_to_input_dir:
        -subdir
            - shapefile.cpg
            - shapefile.dbf
            - shapefile.prj
            - shapefile.shp
            - shapefile.shx
        -subdir2
        -subdir3
"""
function read_shpfile(path_to_input_dir, subdir)
    path_to_shpfile = joinpath(path_to_input_dir, subdir)
    files = readdir(path_to_shpfile)
    file = String
    for f in files
        if occursin(".shp", f)
            file = joinpath(path_to_shpfile, f)
            break
        end
    end
    gdf = GeoDataFrames.read(file)
    return gdf
end

function read_json_geodataframe(path_to_input_dir, subdir)
    path_to_shpfile = joinpath(path_to_input_dir, subdir)
    files = readdir(path_to_shpfile)
    file = String
    for f in files
        if occursin(".json", f)
            file = joinpath(path_to_shpfile, f)
            break
        end
    end
    gdf = GeoDataFrames.read(file)
    return gdf
end

"""
    reading in CSV file at pah "f"
"""
function read_csv(f::String)
    df = DataFrame(CSV.File(f; normalizenames=true))
    return df
end

"""
    p_training_agents(step, p_max, p_min, n_steps)
function to determine the percentage of training agents at each step.
linearly increases from p_min to p_max over n_steps
similar to epsilon decay, but increasing
"""
function p_training_agents(step::Int64, p_max::Float64, p_min::Float64, n_steps::Int64)
    m = (p_max-p_min)/(n_steps)
    y = (m*step)+p_min
    (y>p_max) && (y=p_max)
    return y
end

"""
    return agent save data
"""
function get_agent_save_data()
    adata = [
            :pos, 
            :pos_guid,
            # :ntwk_num,
            :family,
            :age,
            :p_migr,
            :p_expd,
            :p_elec,
            :p_trns,
            :action,
            :reward,
            :q_ntng,
            :q_leav,
            :q_elev,
            :q_gnrt,
            # :percent_year_exposed,
            ]
    return adata
end


"""
    return model save data
"""
function get_model_save_data()
    mdata = [
            :year,

            :n_occupied,
            :n_unoccupied,
            
            :n_elevated_occupied,
            :n_elevated_unoccupied,

            :n_generator_occupied,
            :n_generator_unoccupied,

            :n_elevated_generator_occupied,
            :n_elevated_generator_unoccupied,
            ]
    return mdata
end

"""
    return space save data
"""
function get_space_save_data()
    sdata = [
            :pos,
            :occupied,
            :elevated,
            :generator,
            :p_expd,
            :p_elec,
            :p_trns,
    ]
    return sdata
end

"""
    makedir(dir::String)
makes directory if it doesn't exist already
"""
function makedir(dir::String)
    if ~isdir(dir)
        mkpath(dir)
    end
end



"""
    write_out(data, model_runname, filename)
writing output dataframe to file
"""
function write_out(data, model, filename)
    fn = joinpath(model.output_dir, filename)
    CSV.write(fn, data)
end

function write_train(data::DataFrame, model::ABM, filename::String)
    fn = joinpath(model.training_dir, filename)
    CSV.write(fn, data)
end

function write_train(data::DataFrame, filename::String, path_to_training_dir::String)
    fn = joinpath(path_to_training_dir, filename)
    CSV.write(fn, data)
end


function alpha_calc(model::ABM, n_alphas::Int64)
    u = Uniform(0,1)                            # setting up uniform distribution
    alphas = rand(abmrng(model), u, n_alphas)   # sampling from uniform
    alphas .= alphas./sum(alphas)               # reassigning values so that sum(alpha)=1
    return alphas
end


function setup_pycall()
    # preparing python script with misc. python operations that will be used
    scriptdir = @__DIR__
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    pycall_jl = pyimport("PythonOperations")
    PYTHON_OPS = pycall_jl.misc_python_ops() # setting up python operations
end



function convert_cols_int!(df)
    for col in names(df)
        if col == "guid"
            continue
        end
        df[!,col] = convert.(Int64,df[!,col])
    end
end




