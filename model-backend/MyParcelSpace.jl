# file for parcel space functions and calculations


"""
    prepare_parcel_df(path_to_input_dir; seed)
Prepares initial parcel dataframe (prcl_df).
Currently reads parcel dataframe shapefile, and computes distance from each
  parcel to the beach.
Returns prcl_df.
"""
function prepare_parcel_df(path_to_input_dir::String; seed::Int64, cell_size::Int64=25, rotate::Bool=false)
    
    # reading in shapefiles
    prcl_df = read_shpfile(path_to_input_dir, "buildings_drs")
    prcl_df = dropmissing(prcl_df, :csv_sector)                                # removing buildings with missing sector information
    prcl_df = filter(row -> row.csv_sector in ["HS1I", "HS2I", "HS3I"], prcl_df) # only consider buildings that are a part of the housing stock (for now)
    
    prcl_df = prcl_df[!, ["strctid", "guid", "geometry", "year_built", 
                          "no_stories", "sq_foot", "appr_bldg", "repl_cst", 
                          "ffe_elev",  "g_elev", "archetype", "arch_wind", 
                          "arch_flood", "arch_sw", "csv_guid", "csv_sector"]]    

    # TODO: Temporary fix for invalid surge-wave archetypes
    invalid_arch_sw = [6,7,8,9,10,12,15,16,18,19]
    for i = 1:size(prcl_df)[1]
       prcl_df[i,:arch_sw] = ifelse(prcl_df[i,:arch_sw] in invalid_arch_sw, 5, prcl_df[i,:arch_sw])
    end

    prcl_df[!,:numprec] = ones(size(prcl_df,1))*2                               # TODO: implement HUA
    prcl_df[!,:row_idx] = 1:size(prcl_df,1)                                     # used for fast dataframe lookup
    prcl_df[!,:occupied] = ones(Int64, size(prcl_df,1))                                # building occupied or not
    prcl_df[!,:elevated] = zeros(Int64, size(prcl_df,1))                               # building elevated or not
    prcl_df[!,:generator] = zeros(Int64, size(prcl_df,1))                              # generator installed or not
    prcl_df[!,:p_expd] = zeros(size(prcl_df,1))                                 # percent of year exposed
    prcl_df[!,:p_elec] = zeros(size(prcl_df,1))                                 # percent of year with electricity loss
    prcl_df[!,:p_trns] = zeros(size(prcl_df,1))                                 # percent of year with increase in travel time
    # prcl_df[!,:row] = collect(1:size(prcl_df,1))

    (rotate==true) && (rotate_gdf!(prcl_df))                                    # rotate buildings to fit in grid better
    add_pos_col!(prcl_df, cell_size)                                            # adding grid position as col
    return prcl_df
end

"""
    add_pos_col!(prcl_df::DataFrame)
add's position column to prcl_df
position column is tuple of (x,y) corresponding to grid cell
each cell is 10x10 meters.
agents.jl grid origin appears to be lower left
"""
function add_pos_col!(prcl_df::DataFrame, cell_size::Int64)
    pos = Tuple[]
    for i = 1:size(prcl_df,1)
        pos_ = (prcl_df[i,:x], prcl_df[i,:y])
        # pos_ = ceil.(Int64, pos_./10)
        pos_ = ceil.(Int64, div.(pos_,cell_size))
        push!(pos, pos_)
    end
    prcl_df[!,:pos] = pos
end
"""
    rotate_gdf(prcl_df::DataFrame, theta, minx, miny)
rotates the points in the geodataframe about (minx, miny) by an angle of theta
theta is in degrees
    todo: rename this function to something more description
"""
function rotate_gdf!(prcl_df::DataFrame, theta=33.0, minx=294400, miny=3219510)
    # first converting to CRS that's in meters
    Lons, Lats = geom_to_xy(prcl_df)                                            # getting longitude and latitude; 
    coords = zip(Lats, Lons)                                                    # zipping lat/long
    df = DataFrame(geometry=ArchGDAL.createpoint.(coords))                      # creating dataframe for conversion below
    df.geometry = ArchGDAL.reproject(df.geometry, GFT.EPSG(4326), GFT.EPSG(32615)) # converting from lat/long to x,y in meters
    x, y = geom_to_xy(df)                                                       # now getting x and y in meters

    dx = x.-minx                # x-distance from origin to each point
    dy = y.-miny                # y-distance from origin to each point
    d = sqrt.(dx.^2 .+ dy.^2)   # distance from origin to each point

    gamma = atan.(dy./dx)        # angle between baseline and line from origin to point
    alpha = gamma.-deg2rad(theta)         # angle between theta line and gamma line

    xp = d.*cos.(alpha)         # new x
    yp = d.*sin.(alpha)         # new y
    coords = zip(xp,yp)         # zipping up new x and y
    prcl_df[!,:geometry] = ArchGDAL.createpoint.(coords) # creating new geometry column
    prcl_df[!,:x] = xp          # saving xp
    prcl_df[!,:y] = yp          # saving yp
end


function geom_to_xy(prcl_df::DataFrame)
    Xs = Float64[]
    Ys = Float64[]
    for g_ in prcl_df.geometry
        p = ArchGDAL.Geometry(g_.ptr)
        x = ArchGDAL.getx(p,0)
        y = ArchGDAL.gety(p,0)
        push!(Xs, ArchGDAL.getx(p,0))
        push!(Ys, ArchGDAL.gety(p,0))
    end
    return Xs, Ys
end

"""
    assign_distance_col!(prcl_df, feature, name::Sting)
computes distance from parcel to feature
creates a column in the parcel dataframe with this distance
prcl_df: parcel dataframe
feature: feature that distance is computed from; GeoDataFrame
name: name of new column that will be created
"""
function assign_distance_col!(prcl_df, feature, name::String)
    dist = Vector{Float64}(undef, size(prcl_df,1))
    prcl_df_geom = prcl_df[:,"geometry"]                                        # getting geometry column
    for i = 1:size(prcl_df,1)
        dist[i] = GeoDataFrames.distance(prcl_df_geom[i], feature[1,:].geometry)
    end
    prcl_df[!, name] = dist
end


"""
    computes distance between two points
"""
function distance_calc(pt1, pt2)
    d = GeoDataFrames.distance(pt1.geometry, pt2.geometry)
end

"""
    get_prcl_by_guid(prcl_df, guid)
function to get parcel in parcel_df by guid; 
returns single row
"""
get_prcl_info(df::DataFrame, pos::Int64, col::String) = df[pos,col]
get_prcl_info(df::DataFrame, pos::Int64, col::Symbol) = df[pos,col]
get_prcl_info(model::ABM, pos::NTuple{2,Int64}, col::String) = model.prcl_df[model.pos_to_guid_idx[pos],col]
get_prcl_info(model::ABM, pos::NTuple{2,Int64}, col::Symbol) = model.prcl_df[model.pos_to_guid_idx[pos],col]
get_prcl_info(model::ABM, pos::NTuple{2,Int64}) = model.prcl_df[model.pos_to_guid_idx[pos],:]
""" 
todo: 
could try to speed up the above operation. it appears that indexing with a
  vector increases the number of allocations
"""

"""
    get_prcl_feature(prcl::DataFrame, key::String)
used to get column value from parcel; prcl here is a dataframe with a single row
"""
# get_prcl_feature(prcl::DataFrame, key::String) = prcl[!,key][1]
# get_prcl_feature(prcl::DataFrame, key::Symbol) = prcl[!,key][1]


"""
    update_prcl_feature(prcl::DataFrame, key::string, value::Float64)
"""
# update_prcl_feature!(model::ABM, guid::String, key::String, value) = model.prcl_df[:,key] .= ifelse.(model.prcl_df[!,"guid"].==guid, value, model.prcl_df[!,key])
# update_prcl_feature!(model::ABM, guid::String, key::Symbol, value) = model.prcl_df[:,key] .= ifelse.(model.prcl_df[!,"guid"].==guid, value, model.prcl_df[!,key])


"""
    guid2idx(guid::String, model::ABM) 
used to get the index associated with the supplied guid
"""
guid2idx(guid::String, model::ABM) = model.prcl_df[only(findall(==(guid), model.prcl_df.guid)), "row_idx"] #[1]
guid2idx(guid::String, prcl_df::DataFrame) = prcl_df[only(findall(==(guid), prcl_df.guid)), "row_idx"] #[1]

"""
    idx2guid(idx::Int64, model::ABM)
used to get the space index associated with the guid 
"""
idx2guid(idx::Int64, model::ABM) =   model.prcl_df[only(findall(==(idx),  model.prcl_df.row_idx)), "guid"]

"""
    next_avail_id(model)
returns the next available id in the model space
"""
function next_avail_id(model::ABM)
    id = maximum(allids(model)) + 1
    return convert(Int, id)
end








