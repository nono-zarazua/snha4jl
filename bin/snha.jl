#!/usr/bin/env julia
# bin/snha.jl

using snha4jl
using DelimitedFiles # (Since your CLI uses readdlm and writedlm [cite: 250, 251])

# ------------ MAIN----------------
const HELP_TEXT="""
    \$ julia MGraph.jl
    Usage: MGraph.jl --help|--create-data|--snha
    Graph parser by Roberto Zarazuanav, 2026
    SNHA algorithms for creating graphs, data, data and association chains
    -----------------------------------------------
    Avaliable subcommands and their options:

    --help - display this help page
    --gnew TYPE - create new directed graph, avialable types are werner or barabasi
            for this option the following other arguments will be considered
                --mode  s - "directed" (default) or "undirected"
                --nodes n - number of nodes, used only for barabasi graphs; default: 10
                --edges n - number of edges, used only for barabasi graphs;default: 15
                -o      s - output csv for adjacency matrix
                -v        - when active, it generates an image from the created graph
    --create-data - Create correlation data for a given graph
            for this option the following other arguments will be considered
                --graph      file - (mandatory) csv file of created graph
                -o           s    - output csv for generated data
                --steps      n    - the number of measurements per node; default: 100
                --iterations n    - the number of iterations; default: 15
    --snha - predict graph structure based on correlation data
            for this option the following arguments are used
                --method     s - analyze the given data file using the method: pearson or spearman; default: pearson
                --alpha      f - significance threshold, default: 0.01
                --threshold  f - correlation threshold, default: 0.01
                --data       s - input .csv containing data
                -v             - when active, it generates an image from the created graph

"""

function main(argv)
    #println(argv)  
    if isempty(argv) || "--help" in argv
        println(HELP_TEXT)
        return
    end
    
    validcmd = ["--gnew","--snha","--create-data"]
    if !(argv[1] in validcmd)
        error("The argument used is not a valid argument from $validcmd")
    elseif argv[1] == "--gnew"
        validtypes = ["werner","barabasi"]
        if length(argv) < 2 || !(lowercase(argv[2]) in validtypes)
            error("The second argument for --gnew must be a valid type of graph in $validtypes")
        end

        type = lowercase(argv[2])
        
        args = Dict{Symbol,Any}()
        valid_args = ["--nodes", "--edges", "-v", "--mode", "-o"]
        v = false
        o = false
        output = "gnew-output.csv"
        i = 3
        while i <= length(argv)
            val = nothing
            if !(argv[i] in valid_args)
                error("Option $(argv[i]) not valid for --gnew. See valid options: $valid_args.\n")
            end

            if (i+1) <= length(argv) && !(argv[i+1] in valid_args)
                val = argv[i+1]
            end

            if argv[i] == "--nodes"
                valint = tryparse(Int, val)
                if isnothing(valint)
                    error("The value for --nodes must be an integer. You provided: $val.")
                end
                args[:nodes] = valint
            elseif argv[i] == "--edges"
                valint = tryparse(Int, val)
                if isnothing(valint)
                    error("The value for --edges must be an integer. You provided: $val.")
                end
                args[:edges] = valint
            elseif argv[i] == "-v" 
                v = true
                i += 1
                continue
            elseif argv[i] == "--mode"
                val = lowercase(val)
                if !(val in ["directed", "undirected"])
                    error("The value for --mode should be 'directed' or 'undirected'. You provided: $val.")
                end
                args[:mode] = val
            elseif argv[i] == "-o"
                o = true
                if isnothing(val)
                    i += 1
                    continue
                else
                    output = val
                end
            end
            i += 2
        end
        g = gnew(;type=type, args...)
        show(stdout, "text/plain", g)
        println()
        println(v)
        if v
            g_dot = graph2dot(g, type=type, mode=args[:mode])
            kroki(g_dot, type="graphviz", cache = false, name="gnew-graph")
        end
        if o
            nrows = names(g,1)
            ncols = names(g,2)
            mat = g.array
            header = reshape(vcat("Nodes",ncols),1,:)
            content = hcat(nrows,mat)
            final = vcat(header,content)
            open(output, "w") do io
                println(io, get(args, :mode, "directed"))
                println(io, type)
                writedlm(io,final,",")
            end
        end
        
    elseif argv[1] == "--create-data"
        if !("--graph" in argv)
            error("A .csv containing an adjacency matrix should be given using --graph.")
        end
        args = Dict{Symbol,Any}()
        valid_args = ["--steps", "--iterations", "--graph", "-o"]
        o = false
        output = "create-data-output.csv"
        filename = nothing
        i = 2
        while i <= length(argv)
            val = nothing
            if !(argv[i] in valid_args)
                error("Option $(argv[i]) not valid for --create-data. See valid options: $valid_args.\n")
            end

            if (i+1) <= length(argv) && !(argv[i+1] in valid_args)
                val = argv[i+1]
            end

            if argv[i] == "--steps"
                valint = tryparse(Int, val)
                if isnothing(valint)
                    error("The value for --steps must be an integer. You provided: $val.")
                end
                args[:n] = valint
            elseif argv[i] == "--iterations"
                valint = tryparse(Int, val)
                if isnothing(valint)
                    error("The value for --iterations must be an integer. You provided: $val.")
                end
                args[:iter] = valint
            elseif argv[i] == "--graph" 
                if isnothing(val)
                    error("Argument --graph requires a filename.")
                end
                if !endswith(val, ".csv")
                    error("Input file must be a .csv file. You provided: $val")
                end
                filename = abspath(val)
                if !isfile(filename)
                    error("File does not exist at: $filename.")
                end
            elseif argv[i] == "-o"
                o = true
                if isnothing(val)
                    i += 1
                    continue
                else
                    output = val
                end
            end
            i += 2
        end
        f = open(filename,"r")
        graph_mode = strip(readline(f))
        type = strip(readline(f))
        close(f)
        data, header = readdlm(filename,',', header=true, skipstart=2)
        cnames = vec(header[1,2:end])
        data = Int64.(data[:,2:end])
        g = NamedArray(data,(cnames,cnames))
        g_dat = graph2data(g; args...)
        if o
            open(output, "w") do io
                final = vcat(permutedims(names(g_dat)), Matrix(g_dat))
                println(io, graph_mode)
                println(io, type)
                writedlm(io,final,",")
            end
        else
            show(stdout, "text/plain", g_dat)
        end

    elseif argv[1] == "--snha"
        if !("--data" in argv)
            error("A .csv containing an adjacency matrix should be given using --data.")
        end
        args = Dict{Symbol,Any}()
        valid_args = ["--data", "--method", "--alpha", "-v", "--threshold"]
        v = false
        filename = nothing
        i = 2
        while i <= length(argv)
            val = nothing
            if !(argv[i] in valid_args)
                error("Option $(argv[i]) not valid for --snha. See valid options: $valid_args.\n")
            end

            if (i+1) <= length(argv) && !(argv[i+1] in valid_args)
                val = argv[i+1]
            end

            if argv[i] == "--method"
                val = lowercase(val)
                valid_methods = ["pearson", "spearman"]
                if !(val in valid_methods)
                    error("Only methods in $valid_methods are supported.")
                else
                    args[:method] = val
                end
            elseif argv[i] == "--alpha"
                valfl = tryparse(Float64, val)
                if isnothing(valfl) || valfl > 1.0 || valfl < 0.0
                    error("The value for --alpha must be an float between 0.00 and 1.00. You provided: $val.")
                end
                args[:alpha] = valfl
            elseif argv[i] == "--threshold"
                valfl = tryparse(Float64, val)
                if isnothing(valfl) || valfl > 1.0 || valfl < 0.0
                    error("The value for --threshold must be an float between 0.00 and 1.00. You provided: $val.")
                end
                args[:threshold] = valfl
            elseif argv[i] == "--data" 
                if isnothing(val)
                    error("Argument --data requires a filename.")
                end
                if !endswith(val, ".csv")
                    error("Input file must be a .csv file. You provided: $val")
                end
                filename = abspath(val)
                if !isfile(filename)
                    error("File does not exist at: $filename.")
                end
            elseif argv[i] == "-v"
                v = true
                i += 1
                continue
            end
            i += 2
        end
        f = open(filename,"r")
        graph_mode = strip(readline(f))
        type = strip(readline(f))
        close(f)
        data, header = readdlm(filename,',', header=true, skipstart=2)
        df = DataFrame(Float64.(data), Symbol.(vec(header[1,1:end])))
        g_snha = asg(df; args...)
        
        if v
            g_dot = graph2dot(g_snha["theta"], type=type, mode=graph_mode)
            kroki(g_dot, type="graphviz", cache=false,name="snha-graph")
        else
            show(stdout, "text/plain", g_snha["theta"])
            println()
        end

    else
        println(HELP_TEXT)
        return
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

    
