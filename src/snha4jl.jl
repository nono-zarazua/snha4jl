# src/snha4jl.jl
module snha4jl

# 1. Load dependencies for the whole package
using NamedArrays, Statistics, StatsBase, Distributions, DataFrames
using CodecZlib, Base64, CRC32c, Downloads, LinearAlgebra, DelimitedFiles
using Random: randperm, rand, shuffle!, shuffle
using CSV

# 2. Include your separate files so they share this module
include("MGraph.jl")
include("datasets.jl")

# 3. Export the functions you want users to be able to call
export load_decathlon
export gnew, d2u, autonames, deg, graph2dot, kroki, graph2data, nodeColors
export asg, centrality, shortest_paths, simple_paths, components, closeness, u2d

end