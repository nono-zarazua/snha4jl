#!/usr/bin/env julia

using CSV
using DataFrames

export load_decathlon

"""
    load_decathlon()

Loads the built-in 1988 Decathlon dataset as a DataFrame.
"""
function load_decathlon()
    # @__DIR__ is the 'src' folder.
    # "..", "data" means "go up one level, then into the data folder"
    filepath = joinpath(@__DIR__, "..", "data", "decathlon88.csv")
    
    return CSV.read(filepath, DataFrame)
end