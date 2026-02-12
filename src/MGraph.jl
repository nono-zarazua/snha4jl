#!/usr/bin/env julia
"""#'
---
title: MGraph.jl
author: Roberto Zarazua
date: 2025-10-28
---
"""

module MGraph
# ════════════════════════════════════════════════════════════════════════════

using NamedArrays, Statistics, StatsBase, Distributions, DataFrames
using CodecZlib, Base64, CRC32c, Downloads, LinearAlgebra, DelimitedFiles
using Random: randperm, rand, shuffle!, shuffle

# ════════════════════════════════════════════════════════════════════════════

"""#'
MGraph — tiny helpers for graphs.

**Table of Contents**

Exports:

* [gnew](#gnew) - build a graph from an adjacency matrix/list or canned generators
* [asg](#asg) - create an association chain graph using the St. Nicolas House Algorithm
* [autonames](#autonames) - create names for nodes and other data structures
* [centrality](#centrality) - centrality measures such as betweenness, closeness, degree, eigenvector und katz centrality
* [components](#components) - extract graph components
* [d2u](#d2u) - create an undirected graph out of a directed graph
* [deg](#deg) - number of undirected or incoming and outgoing edges
* [edgeDirShuffle](#edgeDirShuffle) - shuffle the direction of edges in an directed graph without multiedges.
* [graph2dot](#graph2dot) - convert a graph to Graphviz DOT
* [graph2data](#graph2data) - create correlation data for the given graph.
* [kroki](#kroki) - visualize diagram code for ditaa, graphviz, etc.
* [simple_paths](#simple_paths) - calculate all simple paths between two nodes
* [nodeColors](#nodeColors) - node colors for directed graphs
* [shortest_paths](#shortest_paths) - calculate the shortest path for a given graph
* [u2d](#u2d) - create an directed graph out of an undirected graph


Helper Functions

* [pmatch](#pmatch) - partial match helper similar to R's pmatch.

```{jl}
println("Hello Julia!")
using .MGraph

```
"""
# ════════════════════════════════════════════════════════════════════════════

# Bring in deps (avoid global using in packages unless needed)
# using SparseArrays
# using LinearAlgebra

# What you want users to see by default:
export gnew, d2u, autonames, deg, graph2dot, kroki, graph2data, nodeColors
export asg, centrality, shortest_paths, simple_paths, components, closeness, u2d

"""#'

## Export Functions

<a name="gnew"> </a>
**gnew(x; mode="directed")**

> Create an new graph object based on either an adjacency matrix.

> This function ...

> Arguments:

> - _x_ - either a adjacency matrix or an adjacency list, if not given a type must be given
  - _type_ - a graph type, one of 'angie', 'band', 'barabasi', 'circle', 'cluster', 'hubs', 'random', 'regular' or 'werner', default: random
  - _mode_ - either 'undirected' or 'directed', default: 'directed'
  - _nodes_ - number of nodes for a given type, default: 10
  - number_ - of edges for a given type, not used for type 'barabasi', default: 12
  - _m_ - number of edges added in each iteration of type is 'barabasi', default: 1
  - _k_ - the degree for a regular graph, or the number of clusters for a cluster graph or the number of hubs for a 
    hubs graph, default: 2
  - _p_ - the probabilty for edges if type is gnp
  - _power_ - the power for preferential attachment if type is 'barabasi', 1 is linear preferential attachment, 
    below one is sublinear, smaller hubs, 0 is no hobs, above 1 is super linear with super hubs, default: 1

> Examples:

```{jl label=gnew}
using NamedArrays
L=string.('A':'Z')[1:7]
M=NamedArray(zeros(Int64,7,7),(L,L),("row", "col"))
M[["A","B"],"C"]=[1,1]
M["C","D"]=1
M["D",["E","F"]]=[1,1]
M["E","F"]=1
G=gnew(M)
R=gnew(type="random",nodes=8,edges=9)
A = gnew(type="angie",nodes=8,edges=9)
B = gnew(type="band",nodes=8)
C = gnew(type="circle",nodes=8)
C
```


"""



function gnew(x::Union{AbstractMatrix,NamedArray,Nothing}=nothing; type="random", mode="directed", nodes=10, edges=12, m=1, k=3, p = nothing, power=1)
    types = ("angie","band","barabasi","circle","cluster","gnp","hubs","random","regular","werner")
    # Check if there is at least a partial match of type to a types[i]
    type = pmatch(type, types)

    # Check if there is an 'x' argument given
    if x !== nothing
        if size(x,1) != size(x,2)
            error("Input x must be a square adjacency matrix when provided")
        elseif isa(x, NamedArray)
                xmat = copy(x)
        else
            nrow = size(x,1)
            # Include node names
            nms = autonames(nrow)
            xmat = NamedArray(x, (nms,nms), ("row","col"))
        end
    # If there is no 'x' argument there should be a 'type' argument
    elseif isnothing(x)
        if type === nothing
            error("Either a matrix or a type must be given")
        end
        
        # Start building adjacency matrix and names
        xmat = zeros(Int64, nodes, nodes)
        nms = autonames(nodes)
        xmat = NamedArray(xmat, (nms,nms), ("row","col"))
        nrow = size(xmat, 1)

        # Random type
        if type == "random"
            # Use top triangle and exclude diagonal to avoid self loops
            # List of all upper-triangular index pairs (i,j)
            upper_indices = [(i,j) for i in 1:nrow, j in 1:nrow if i < j]
            # Rnadomly select edges
            sampled = randperm(length(upper_indices))[1:min(edges, length(upper_indices))]

            # Create edge from i to j from sampled
            for s in sampled
                i, j = upper_indices[s]
                xmat[i,j] = 1
            end
            # Randomly flip edge directions
            xmat = edgeDirShuffle(xmat)
        elseif type in ("band", "circle")
            for i in 1:nrow-1
                xmat[i,i+1] = 1
            end
            if type == "circle"
                xmat[nrow,1] = 1
            end
        elseif type == "barabasi"
            xmat[2,1] = 1
            for n in 3:nrow
                if m == n
                    sel = n-1
                else
                    sel = m
                end
                d = deg(xmat,mode=mode).array .^power
                # preferential attachment to nodes with higher degree
                idx = sample(1:(n-1), Weights(d[1:(n-1)]), sel; replace=false)
                xmat.array[idx,n] .= 1
            end
            xmat = edgeDirShuffle(xmat)
        elseif type == "gnp"
            if isnothing(p)
                error("For graphs of type 'gnp' you must give the edge probabilty 'p'!")
            end
            if p < 0 || p > 1
                error("p must be withing 0 and 1!")
            end
            b = Bernoulli(p)
            xmat.array .= rand(b,nrow,nrow)

            for i in 1:nrow
                xmat[i,i] = 0
            end
        elseif type =="angie"
            done = [nms[1]]
            nds = nms[2:end]
            while !isempty(nds)
                tar = sample(done,1)[1]
                xmat[tar,nds[1]] = 1
                push!(done,nds[1])
                nds = nds[2:end]
            end
            upper_indices = [CartesianIndex(i,j) for i in 1:nrow, j in 1:nrow if i < j]
            while sum(xmat) < edges
                idx = findall(xmat[upper_indices] .== 0)
                isempty(idx) && break
                pos = rand(idx)
                xmat[upper_indices[pos]] = 1
            end
            xmat = edgeDirShuffle(xmat)
        elseif type =="werner"
            xmat = xmat[1:6,1:6]
            xmat[[1,2],3] .= 1
            xmat[3,4] = 1
            xmat[4,5:6] .= 1
            xmat[5,6] = 1
        elseif type in ("cluster","hubs")
            nsize = nodes ÷ k
            rem = nodes % k
            esize = edges ÷ k
            ree = edges % k
            pos = 0
            for i in 1:k
                s = nsize
                e = esize
                if rem > 0
                    s = nsize+1
                    rem = rem - 1
                end
                if ree > 0
                    e = e + 1
                    ree = ree -1
                end
                if type == "cluster"
                    g=gnew(type="angie",nodes=s,edges=e)
                else
                    g=transpose(reshape([0; ones(s-1); zeros(s*s-s)], s, :))
                end
                xmat[(pos+1):(pos+size(g,1)),(pos+1):(pos+size(g,1))]=g
                pos+=s
            end
            xmat=edgeDirShuffle(xmat)
        elseif type=="regular"
            if k in (1,3)
                if nodes % 2 != 0
                    error("Regular graphs with k = 1 or k = 3 must have an even number of nodes")
                end
            end
            if k == 1
                for i in 1:2:(size(xmat,1)-1)
                    xmat[i,i+1] = 1
                end
            elseif k <=3
                xmat = gnew(type="circle",nodes=nodes)
                if k == 3
                    half_nodes = div(nodes, 2)
                    for i in 1:half_nodes
                        xmat[i,i+half_nodes] = 1
                    end
                end
            else
                error("Error: Only values of k <= 3 are implemented for regular graphs")
            end
        end
    else
        error("Either a matrix or a type must be given")
    end
        
    if mode == "undirected"
        xmat = d2u(xmat)
    end

    return(xmat)
end



"""#'

<a name="asg"> </a>
**asg(data,method="pearson")** 

> Create an association chain graph.

> This function returns graphs which are build from association chains.
  For details on the algorithm see [Groth et al. 2019](https://doi.org/10.1127/anthranz/2019/1027) and [Hermanussen et. al. 2021](https://doi.org/10.3390/ijerph18041741).

> Arguments:

> - _data_  - a dataframe where network nodes are the rownames and data variables are in the columns
  - _alpha_ - confidence threshold for p-value edge cutting after all chains were generated, default: 0.01
  - _method_ - method to calculate correlation/association values, can be pearson, spearman, kendall, cor.fk (requires pcaPP package), rpart or mi (mutual information) default: 'pearson'.
  - _threshold_ - correlation coefficient threshold which r values should be used for chain generation, default: 0.1
  - _check.singles_ - should isolated nodes connected with sufficent high R^2 and significance, default: false
  - _chains.clean_ - should shorter chains be removed if they are in longer chains, and should reverse dubplicated chains be removed, default: true
  - _prob_ - should probabilities be computed for each edge using bootstrapping. Only in this case the parameters starting with prob are used, default: false
  - _prob.threshold_ - threshold to set an edge, a value of 0.5 means, that the edge must be found in 50% of all samplings, default: 0.2
  - _prob.n_ - number of boostrap samples to be taken, default: 25

> Returns:  An asg graph data object with the fields theta for the adjacency matrix, sigma for the correlation matrix, chains for the association chains and data representing the input data.

> Examples:

> ```{jl label=asg,fig=true,fig.width=7,fig.height=3}
data(swiss)
swg=asg(swiss,method='spearman')
names(swg)
swg["theta"]
round(sigma,2)
# resampling approach
swg=asg(swiss,method='spearman',check.singles=true,prob=true)
swg["theta"]
swg["probabilities"]
> ```

"""

function asg(data::DataFrame;alpha=0.01,method="pearson",threshold=0.01,
                    pcor=false,check_singles=false,prob=false,
                    prob_threshold=0.2,prob_n=25)
    t1 = time()
    nrow,ncol = size(data)
    if prob
        if nrow == ncol
            dmat = Matrix(data)
            has_one_diag = all(x -> isapprox(x, 1.0, atol=1e-5), [dmat[i,i] for i in 1:nrow])
            is_symmetric = isapprox(dmat, transpose(dmat), atol=1e-5)
            if has_one_diag && is_symmetric
                error("It looks like you passed a correlation matrix. Bootstrapping (prob=true) requires RAW DATA (rows=samples).")
            end
        end

        as = asg(data,alpha=alpha,method=method,threshold=threshold,pcor=pcor,
                check_singles=check_singles,prob=false)
        prob_sum = NamedArray(zeros(Float64, ncols, ncols), (names(data), names(data)), ("row", "col"))
        rand_prob = NamedArray(zeros(Float64, ncols, ncols), (names(data), names(data)), ("row", "col"))
        for i in 1:prob_n
            sam = rand(1:nrow, nrow)
            asi = asg(data[sam,:],alpha=alpha,method=method,threshold=threshold,
                    check_singles=check_singles,prob=false)
        
            prob_sum .+= asi["theta"]
            rand_data = copy(data)
    
            for col in names(data)
                rand_data[!,col] = shuffle(rand_data[!,col])
            end
            asr = asg(rand_data,alpha=alpha,method=method,threshold=threshold,
                check_singles=check_singles,prob=false)
            rand_prob .+= asr["theta"]
        end
        as["probabilities"] = prob_sum ./ prob_n
        rand_prob = vec(rand_prob ./ prob_n)
        as["rand_probabilities"] = copy(as["probabilities"])
        as["rand_probabilities"][:] = rand_prob
        as["theta"] = [as["probabilities"]] .> prob_threshold ? 1.0 : 0.0
        as["p_values"] = copy(as["theta"])
        vprob = vec(as["probabilities"])
        new_p_vals = map(x -> 1.0 - count(r -> x > r, rand_prob) / length(rand_prob), vprob)
        as["p_values"][:] = new_p_vals
    else
        as = data2chainGraph(data,alpha=alpha,method=method,threshold=threshold)
        if pcor
            for i in 1:3
                as["theta"] = pcheck(as)
            end
        end
        if check_singles
            colnames = names(data)
            if method == "pearson"
                cmt = NamedArray(Statistics.cor(Matrix(data)),(colnames,colnames),("row","col"))
                cmt = cmt.^2
            elseif method == "kendall"
                cmt = NamedArray(StatsBase.corkendall(Matrix(data)),(colnames,colnames),("row","col"))
                cmt = abs(cmt)
            elseif method == "spearman"
                cmt = NamedArray(StatsBase.corspearman(Matrix(data)),(colnames,colnames),("row","col"))
                cmt = cmt.^2
            else
                error("Method not yet implemented.")
            end
            [cmt[i,i] = 0.0 for i in 1:size(cmt,1)];
            idx = findall(x -> x == 0, vec(sum(as["theta"], dims=2)))
            for i in idx
                max_val = maximum(cmt[i,:])
                if  max_val > threshold
                    j = findfirst(x -> x == max_val, cmt[i, :])
                    if !isnothing(j)
                        pair_data = data[:,[colnames[i],colnames[j]]]
                        res = corTest(pair_data, method=method)
                        p_val = res["p_value"][1,2]
                        if p_val < alpha
                            as["theta"][i,j] = 1
                            as["theta"][j,i] = 1
                        end
                    end
                end
            end
        end
        as["probabilities"] = as["theta"]
    end
    as=ReduceChains(as)
    return as
end

"""#'

<a name="autonames"> </a>
**autonames(n, prefix)**
 
> Create names for nodes and other data structures.
 
> This function aids in creating standard node labels for graphs.
 
> Arguments:

>  - _prefix_ - one or more prefixes to be used when labels > 26
>   - _n_ - how many labels

> Examples:
 
> ```{jl label=autonames}
  print(autonames(12))
  print(autonames(12,LETTERS[1:4]))
  print(autonames(12,"R"))
> ```



"""
function autonames(n::Integer; prefix::Union{Nothing,AbstractString,AbstractVector{<:AbstractString}}=nothing)
    n <= 0 && return String[]

    if prefix === nothing
        if n <= 26
            return string.('A':'Z')[1:n]
        elseif n <= 26*9
            return autonames(n,prefix=string.('A':'Z'))
        else
            return autonames(n,prefix="N")
        end
    else
        # Normalize prefix to a vector of strings
        nms = prefix isa AbstractString ? [String(prefix)] : String.(prefix)
        L = length(nms)
        L == 0 && throw(ArgumentError("prefix vector must be non-empty"))  
        ln = cld(n, L)
        frmt = ndigits(ln)

        nms_tmp = Vector{String}(undef,n)
        for k in 1:n
            i = ((k-1) % L) + 1
            grp = ((k-1) ÷ L) + 1
            nms_tmp[k] = nms[i] * lpad(string(grp), frmt, '0')
        end
        return nms_tmp
    end
end

"""

<a name="centrality"> </a>
**centrality(g,method="eigen")** 

> Return various centrality measures such as betweenness, closeness, degree, eigenvector or katz centrality.

> This function returns centrality values for the nodes of a network.
  Please note that eigenvector and closeness centrality have problems with graphs where not all
  nodes are in a single component. For directed graph you should either use method 'pagerank' or degree with the mode='in' as argument. The 'pagerank' algorithm used here is similar to the 'power' algorithm in the igraph method page_rank with the argument algo='power'. 

> Arguments:

> - _g_ - an adjacency matrix
  - _method_ - the centrality measure, possible values are 'closeness','degree', 'eigen' 'harmonic' ([Rochat 2009](http://infoscience.epfl.ch/record/200525/files/%5bEN%5dASNA09.pdf)) or 'katz' (Katz 1953) currently, default: 'katz'
  - _alpha_ - the alpha value for katz centrality, should be between 0 and 0.2, default: 0.1
  - _d_ - the dumping factor for method 'pagerank', default: 0.85
  _ _iter.max_ - the maximum number of iterations for method 'pagerank', default: 100
  - _norm_ - for betwenness Bn=(2*B)/(n*n-3*n+2), closeness, degree and katz centrality, should centrality normed between 0 (katz value=1) and 1 (max katz) for katz, by multiplication with _n-1_ for closeness and by dividing by |V|-1 by degree centrality default: FALSE
  - _..._ - arguments delegated to the degree function
#'
> Examples:

"""

function centrality(g::Union{Matrix,NamedArray};method="eigen",alpha=0.1,norm=false,iter_max=100,d=0.85,kwargs...)
    if g != g'
        cmat = d2u(g)
    else
        cmat = g
    end
    methods = ("betweenness","closeness","degree","eigen","harmonic","prank","katz")
    method = pmatch(method,methods)
    if method == "betweenness"
        res = betweenness(cmat)
        if norm
            n = size(cmat,1)
            res = 2 .* res ./ (n*n-3*n+2)
        end
        return res
    elseif method == "closeness"
        res = closeness(cmat,norm=norm)
        return res
    elseif method == "degree"
        d = deg(cmat; kwargs...)
        if norm
            d ./= (length(d) -1)
        end
        return d
    elseif method == "eigen"
        res = abs.(eigen(cmat).vectors[:,end])
        res ./= maximum(res)
        nms = NamedArrays.names(cmat,2)
        res = NamedArray(res,(nms,),("des",))
        return res
    elseif method == "harmonic"
        res = harmonic(cmat,norm=norm)
        return res
    elseif method == "prank"
        res = pagerank(cmat,d=d,iter_max=iter_max)
        return res
    elseif method == "katz"
        res = katzcent(cmat,alpha=alpha,norm=norm)
        return res
    else
        error("Error: Currently only 'betweenness', 'closeness', 'degree', 'eigen','harmonic', 'prank', or 'katz' are supported!")
    end
end

"""#'
<a name="components"> </a>
**components(g)** 

> Return graph components, nodes which are connected are within the same component.

> This function ...

> Arguments:

> - _g_ - anadjacency matrix
> Examples:

> ```{jl label=components}
  G # node G is single
  components(G)
> ```

"""

function components(g::Union{AbstractMatrix,NamedArray})
    if g isa NamedArray
        nms = NamedArrays.names(g,2)
        A = g.array + g.array'
        A = NamedArray(A,(nms,nms),("row","col"))
    else
        nms = autonames(size(g,1))
        A = g + g'
        A = NamedArray(A,(nms,nms),("row","col"))
    end
    A[A.>0] .= 1
    comp = Dict{String, Int}()
    P = shortest_paths(A)
    ri = Set(1:length(nms))
    x = 1
    while !isempty(ri)
        n = first(ri)
        idx = findall(P[:,n] .< Inf)
        for i in idx
            name = nms[i]
            comp[name] = x
        end        
        setdiff!(ri, idx)
        x += 1
    end
    ordered_ids = [comp[name] for name in nms]

    return NamedArray(ordered_ids, (nms,), ("Component",)) 
end

"""#'

<a name="d2u"> </a>
**d2u(g)**
 
> Create an undirected graph out of a directed graph with the same number of edges.
 
> This function gets an directed graph and convertes all edges from directed ones to undirected ones.

> The number of edges should stay the same, the edge sign (+ or -) stays the same.
 
> Arguments:
 
>  - _g_ - a mgraph object or an adjacency matrix

> Examples:
 
> ```{jl label=d2u}
  A=gnew(type="angie",nodes=4,edges=4)
  U=d2u(A)
  print(A)
  print(U)
> ```

"""

function d2u(x)
    nms = NamedArrays.names(x,1)
    b = x.array + transpose(x.array)
    b[b .> 0] .= 1
    b[b .< 0] .= -1
    return NamedArray(b, (nms,nms), ("row","col"))
end


"""#'
    
<a name="deg"> </a>
**deg(g,mode="undirected")**

> Return the number of undirected or incoming and outgoing edges.
 
> This function returns degree centrality measure. For other centrality measures for single nodes
 
> Arguments:
 
>  - _g_ - a mgraph object or an adjacency matrix
   - _mode_ - either 'undirected', 'out' or 'in', default: 'undirected'

> Examples:
 
> ```{jl label=deg}
  A=gnew(type="regular",nodes=8,k=3)
  print(A)
  print(d2u(A)) 
  print(deg(A))
  print(deg(A))
  print(deg(A))
> ```



"""

function deg(x;mode="undirected")
    n = size(x,1)
    nms = x isa NamedArray ? names(x,1) : autonames(n)
    xmat = x isa NamedArray ? x : NamedArray(x,(nms,nms), ("row","col"))
    if isempty(nms) || all(isnothing, nms)
        nms = autonames(n)
        setnames!(xmat, nms, 1)
        setnames!(xmat, nms, 2)
        setdinames!(xmat,("row","col"))
    end
    if mode == "undirected"
        xmat=d2u(xmat)
    elseif mode == "in"
        xmat=transpose(xmat)
    end
    xmat .= Int.(xmat .!= 0)
    d = NamedArray(dropdims(sum(xmat; dims=2), dims=2), (nms,), ("Node",))
    return(d)
end

"""#'

<a name="edgeDirShuffle"> </a>
**edgeDirShuffle(x)**
 
> Shuffle the direction of edges in an directed graph without multiedges.
 
> This function allow to randomize the direction of the edges in a directed graph.

>The total number of edges remains constant. The nodes which are connected remain the same.
 
> Arguments:
 
>  - _x_ - an adjacency matrix.

> Returns: - a mgraph object with shuffled edge directions

> Examples:

> ```{jl label=edgeDirShuffle}
  A=gnew(type="angie",nodes=6,edges=8)
  B=edgeDirShuffle(A)
  print(A)
  print(B)
> ```


"""

function edgeDirShuffle(x::NamedArray{Int})
    if size(x,1) != size(x,2) 
        error("Adjacency matrices are square matrices")
    end
    n = size(x,1)
    s = copy(x)
    for i in 1:n-1
        for j in i+1:n
            w = s[i,j] + s[j,i]
            if w > 0
                if rand(Bool)
                    s[i,j] = w
                    s[j,i] = 0
                else
                    s[i,j] = 0
                    s[j,i] = w
                end
            end
        end
    end
    return(s)
end


"""'#

<a name="graph2data"> </a>
**graph2data(g,n=100)** 

> Create correlation data for the given graph.

> This function is a short implementation of the algorithm in Novine et. al. 2022.

> Arguments:

> - _g_ - adjacency matrix
  - _n_ - the number of measurements per node, default: 100
  - _iter_ - the number of iterations, default: 15
  - _sd_ - initial standard deviation, default: 2
  - _val_ - initial node value, default: 100
  - _prop_ - proportion of the target node value take from the source node, default: 0.05
  - _noise_ - the sd for the noise value added after each iteration using rnorm function with mean 0, default: 1

> Examples:

> ```{jl label=graph2data}
  A=gnew(type="angie",nodes=8,edges=10)
  C=graph2data(A,prop=0.05)
  round(cor(t(C)),2)
  corrplot(cor(t(C)))
> ```

> Here an example which shows that with increasing number of iterations the correlations between the nodes increase.

> ```{jl label=graph2data2,fig=true,fig.cap="",fig.width=9,fig.height=9}
  par(mfrow=c(2,2),mai=rep(0.1,4))
  W=gnew(type="werner")
  lay=matrix(c(1,1,1,3,2,2,3,2,4,3,4,1),ncol=2,byrow=true)
  plot(W,layout=lay)
  for (i in c(15,30,50)) {
     data=graph2data(W,n=200,iter=i)
     as=asg(t(data),method="spearman",alpha=0.1)
     plot(as,layout=lay,edge.text=round(sigma,2))
     text(2.5,1.5,paste("iter =",i),cex=2)
  }
> ```

"""

function graph2data(g;n=100,iter=15,val=100,sd=2,prop=0.05,noise=1)
    if !(g isa NamedArray)
        nms = autonames(size(g,1))
        xmat = NamedArray(g, (nms,nms),("row","col"))
    else
        xmat = copy(g)
        nms = NamedArrays.names(xmat, 1)
    end
    res=zeros(Float64,size(xmat,1),n)
    for i in 1:n
        units=NamedArray(val .+ sd .* randn(size(xmat,1)),nms)
        for j in 1:iter
            for node in shuffle(nms)
                targets=NamedArrays.names(xmat,2)[xmat[node,:].!=0]
                for target in shuffle(targets)
                    P = abs(xmat[node,target])
                    nval = units[node]*(prop*P)
                    nval = nval + units[target]*(1-(prop*P))
                    if xmat[node,target] < 0
                        diff = nval - units[target]
                        nval = units[target] - diff
                    end
                    units[target] = nval
                end
            end
            units = units .+ rand(Normal(0,noise),length(units))
        end
        res[:,i] = units
    end
    df = DataFrame()
    for i in 1:length(nms)
        df[!,nms[i]] = res[i,:]
    end 
    return df
end

"""#'
<a name="graph2dot"> </a>
**graph2dot(g,type="digraph")** 

> Convert a graph to a Graphviz dot representation.

> This gets a graph or adjacency matrix and creates dot graph representation which can used to create
  diagram images using the kroki diagram services using the [kroki](#kroki) function.

> Arguments:

> - _g_ - graph or adjacency matrix
  - _mode_- directed or undirected
  -
  - _type_ - specific type of graph layout
  -_shape_ - graphviz node shape
  -_style_ - graphviz node style
  -_fillcolor_ - graphviz node color
> Examples:

> ```{jl label=graph2dot}
  using NamedArrays
  L=string.('A':'Z')[1:3]
  M=NamedArray(zeros(Int64,3,3),(L,L),("row", "col"))
  M["A","B"]=1
  M["B","C"]=1
  M
  g=graph2dot(M)
  url=kroki(g,type="graphviz")
  print(url)
> ```

> In one line directly within the Markdown text could write.

> ```
  ![ ](` jl kroki(graph2dot(M),type="graphviz")`)
> ```

> ![ ](`jl kroki(graph2dot(M),type="graphviz")`)

> Here an undirected graph example:

> ```
 # next three lines should be on one line
 ![ ](` jl kroki(graph2dot(
     gnew(type="angie",mode="undirected"),type="graph",fillcolor="skyblue"),
     type="graphviz")`)
> ```

> ![ ](`jl kroki(
          graph2dot(
          new(type="angie",mode="undirected"),
          type="graph",fillcolor="skyblue"),
          type="graphviz")`)



"""


function graph2dot(x;  mode="directed", type="custom", custype::Union{String, Nothing}=nothing,shape="circle", style="filled", fillcolor="salmon", inout_color=["salmon","grey80","skyblue"])
    # Define Layout Strategies
    # "layout" determines the engine (dot, neato, circo, etc.)
    # "rankdir" only applies to the 'dot' engine.
    layouts = Dict(
        "circle"    => "layout=circo;",
        "regular"   => "layout=circo;",
        "gnp"       => "layout=fdp; overlap=scalexy; sep=1.5; splines=true;",
        "random"    => "layout=fdp; overlap=scalexy; sep=0.5; splines=true;",
        "cluster"   => "layout=fdp; overlap=scalexy; sep=1.5;",    # fdp is great for clusters
        "barabasi"  => "layout=dot; rankdir=LR;", # or layout=twopi for radial
        "band"      => "layout=circo;",
        "angie"     => "layout=dot; rankdir=LR;",
        "werner"    => "layout=fdp;",
        "hubs"      => "layout=twopi; overlap=false;",  # Radial layout usually looks cool for hubs
        "custom"    => custype
    )
    
    if type == "custom" && isnothing(custype)
        error("Error: A custom layout string must be given when type='custom'.")
    end
    
    config = layouts[type]

    if !(x isa NamedArray)
        nms = autonames(size(x,1))
        xmat = NamedArray(x, (nms,nms),("row","col"))
    else
        xmat = copy(x)
        nms = NamedArrays.names(xmat, 1)
    end

    str = ""
    if mode == "directed"
        start = "digraph"
        if fillcolor == "inout"
            node_style = "node [shape=$shape, style=$style];\n"
            colors = nodeColors(xmat,col=inout_color)
            if type == "werner"
                str *= """A [pos="0,2!", fillcolor=$(colors[1])];\n     
                    B [pos="0,-0!", fillcolor=$(colors[2])];\n  
                    C [pos="1,1!", fillcolor=$(colors[3])];\n  
                    D [pos="2.5,1!", fillcolor=$(colors[4])];\n  
                    E [pos="3.5,2!", fillcolor=$(colors[5])];\n  
                    F [pos="3.5,0!", fillcolor=$(colors[6])];\n"""
            else
                for i in axes(colors,1)
                    str *= "$(nms[i]) [fillcolor=$(colors[i])];\n"
                end
            end
        else
            node_style = "node [shape=$shape, style=$style, fillcolor=$fillcolor];\n"
            if type == "werner"
                str *= """A [pos="0,2!"];\n     
                    B [pos="0,-0!"];\n  
                    C [pos="1,1!"];\n  
                    D [pos="2.5,1!"];\n  
                    E [pos="3.5,2!"];\n  
                    F [pos="3.5,0!"];\n"""
            elseif type == "band"
                str *= """ultimo [fillcolor=transparent, color=transparent, fontcolor=transparent];\n
                        $(nms[end]) -> ultimo [color=transparent];\n
                        ultimo -> $(nms[1]) [color=transparent];\n
                """
            end
        end
    else
        start = "graph"
        node_style = "node [shape=$shape, style=$style, fillcolor=$fillcolor];\n"
        if type == "werner"
            str *= """A [pos="0,2!"];\n     
                    B [pos="0,-0!"];\n  
                    C [pos="1,1!"];\n  
                    D [pos="2.5,1!"];\n  
                    E [pos="3.5,2!"];\n  
                    F [pos="3.5,0!"];\n"""
        elseif type == "band"
            str *= """ultimo [fillcolor=transparent, color=transparent, fontcolor=transparent];\n
                    $(nms[end]) -- ultimo [color=transparent];\n
                    ultimo -- $(nms[1]) [color=transparent];\n
            """
        end
    end

    for i in 1:(size(xmat,2)-1)
        for j in (i+1):size(xmat,2)
            # Edge i -> j
            if xmat[i,j] == 1
                if mode == "directed"
                    str *= "$(nms[i]) -> $(nms[j]);\n"
                else
                    str *= "$(nms[i]) -- $(nms[j]);\n"
                end
            end
            
            # Edge j -> i (Only needed for directed, or if matrix is asymmetric)
            if xmat[j,i] == 1
                if mode == "directed"
                    str *= "$(nms[j]) -> $(nms[i]);\n"
                end
            end
        end
    end

    header = "$start g {\n$config\n"
    
    return header * node_style * str * "\n}\n"
end


"""#'

<a name="kroki"> </a>
**kroki(text,type="ditaa",ext="png")** 
 
> Create diagrams using the online tool [kroki](https://kroki.io).
 
> This function is creates a URL which can be easily embedded into Markdown code for displaying
  diagrams supported by the online tool [kroki.io](https://kroki.io).
  There is as well an online diagram editor, see here [niolesk](https://niolesk.top/).

> Arguments:

> - _text_ - some diagram code, default: "A --> B"
  - _filename_ - some input file, either _text_ or _file_ must be given, default: nothing
  - _type_ - diagram type, supported is ditaa, graphviz, and many others, see the kroki website, default: ditaa
  - _ext_ - file extension, usally 'png', 'svg' or 'pdf', default: 'png'
  - _cache_ - should the image be cached locally using crc32 digest files in an _img_ folder, default: true

> Examples:

> ```{jl label=kroki}
  url1=kroki("
  digraph g { 
     rankdir=\"LR\";
     node[style=filled,fillcolor=salmon];
     A -> B -> C ; 
  }",
  type="graphviz")
  url2=kroki("
  +---------+    +--------+
  |    AcEEF+--->+   BcFEE+
  +---------+    +--------+
  ")
> ```

> To embed the image you can use Markdown code like here:

> ```
  # remove space before r letter
   ![ ](` jl url1 `)
> ```

> Here the output:

> ![ ](`jl url1`)

> And here the image for the second diagram, a Ditaa diagram:

> ![ ](`jl url2`)

> The diagram code can be read as well from a file here a Ditaa file:

> ```{jl}
  url3=kroki(filename="hw.ditaa")
> ```

> ![](`jl url3`)

> For images stored in files it can be even easier without intermediate variables like this:

> ```
  ![ ](`jl kroki(filename="hw.ditaa")`)
> ```

> Here another example:


> ```{jl}
kroki(graph2dot(gnew(type="werner"),type="digraph"),type="graphviz")
> ```

> See also:

> - [graph2dot](#graph2dot) - converting graphs to dot code which can be then send to the kroki service
  - [ditaa documentation I](http://ditaa.sourceforge.net/)
  - [ditaa documentation II](https://github.com/stathissideris/ditaa)
  - [svgbob documentation](https://ivanceras.github.io/content/Svgbob.html)
  - [plantuml documention](https://plantuml.com/)
  - [graphviz documentation](https://graphviz.org/documentation/)
  - [kroki documentation](https://kroki.io/)
  - [niolesk editor](https://niolesk.top/#https://kroki.io/plantuml/svg/eNplj0FvwjAMhe_5FVZP40CgaNMuUGkcdttp3Kc0NSVq4lRxGNKm_fe1HULuuD37-bOfuXPUm2QChEjRnlIMCDmdUfHNSYY6xh42a9Fsegflk-yYlOLlcHK2I2SGtX4WZm9sZ1o8uOzxxbuWAlIGj8cshs6M1jDuY2owyU2P8jAezdnn10j53X0hlBsZFW021Pq7HaVSNw-KN-OogG8F8BAGqT8dXhZjxW4cyJEW6kcC-yHWFagHqW0MfaThhYmaVyE26P_x27qaDmXeruqqAMMw1h-ZlRI4aF3dX7hOwm5XzfIKDctlNcshPT1tFa8JPYAj-Zf5F065sqM=)



"""


function kroki(text="A --> B"; filename=nothing, type="ditaa", ext="png", cache=true, name=nothing)
    if !isnothing(filename)
        if !isfile(filename)
            error("Error: file " * filename * " does not exist!")
        else
            text = read(filename, String)
        end
    end

    data = transcode(ZlibCompressor, codeunits(text))
    b64string = base64encode(data)
    safe_b64 = replace(b64string, "+" => "-", "/" => "_", "=" => "")
    url = "https://kroki.io/$type/$ext/$safe_b64"

    if !isdir("img")
        mkdir("img")
    end

    if isnothing(name)
        chk = string(crc32c(url), base=16)
        out_filename = "$(chk).$(ext)"
    else
        out_filename = "$(name).$(ext)"
    end
    imgname = joinpath("img", out_filename)

    if !isfile(imgname) || !cache
        print("Downloading $(out_filename)...\n")
        try
            Downloads.download(url, imgname)
        catch e
            rm(imgname, force=true) 
            error("Failed to download image. Check syntax or internet.")
        end
    end

    return imgname
end

"""#'
<a name="nodeColors"> </a>
**nodeColors(g,colors=c("skyblue","grey80","salmon")** 

> Returns node colors for directed graphs.

> This function simplifies automatic color coding of nodes for directed graphs.
  Nodes will be colored based on their degree properties, based
  on their incoming and outcoming edges.

> Arguments:

> - _g_ - an adjacency matrix
  - _col_ - default colors for nodes with only incoming, in- and outgoing and only outgoing edges, default: c("skyblue","grey80","salmon")

> Examples:

> ```{jl label=nodeColors}
  A=gnew(type="random",nodes=6,edges=8)
  cols=nodeColors(A)
  degree(A,mode="in")
  degree(A,mode="out")
  cols
  plot(A, layout="star")
  plot(A, layout="star",vertex.color=cols) 
> ```
"""

function nodeColors(g;col=["skyblue","grey80","salmon"])
    colors = fill(col[2],size(g,1))
    out = deg(g,mode="out")
    inc = deg(g,mode="in")
    sources = (out .> 0) .& (inc .== 0)
    sinks = (out .== 0) .& (inc .> 0)
    colors[sources] .= col[1]
    colors[sinks] .= col[3]
    return colors
end

"""#'
<a name="shortest_paths"> </a>
**shortest_paths(g)** 

> Calculate the shortest paths between all graph nodes.

> This function ...

> Arguments:

> - _g_ - an adjacency matrix created with _new_.
 
> Returns: a matrix with the shortest paths

> Examples:

> ```{jl label=spath}
  G
  shortest_paths(G)
  shortest_paths(G,mode="undirected")
> ```
#'
"""

function shortest_paths(g::Union{NamedArray,AbstractMatrix};mode="directed")
    rows,cols = size(g)
    if rows != cols error("Input g must be a square adjacency matrix when provided") end
    if isa(g, NamedArray)
        A = copy(g)
    else
        nms = autonames(rows)
        A = NamedArray(g,(nms,nms),("row","col"))
    end
    
    if mode == "undirected"
        A = d2u(A)
    end
    N = size(A, 1)
    S = NamedArray(fill(Inf, N, N), (names(g,1), names(g,2)), ("rows", "cols"))
    
    S[LinearAlgebra.diagind(S)] .= 0.0
    
    S[A .!= 0] .= 1.0
    # Floyd-Warshall Algorithm (The Standard Logic)
    for k in axes(S,1)
        for i in axes(S,1)
            for j in axes(S,1)
                if S[i,k] != Inf && S[k,j] != Inf
                    new_dist = S[i,k] + S[k,j]
                    if new_dist < S[i,j]
                        S[i,j] = new_dist
                    end
                end
            end
        end
    end
    
    return S
end

"""#'
<a name="simple.paths"> </a>
simple.paths(g)** 

> Return all simple paths between two nodes.

> This function calculates all existing simple paths.  
  A path is simple if no vertice is visited more than once.

> Arguments:

> - _g_ - an adjacency matrix
  - _start_ - the start node 
  - _end_ - the end node
  - _shortest_ - should only the shortest paths(s) be returned, default: FALSE
 
> Returns: a list with all simple paths or only with the shortest path(s) if the argument _shortest_ is set to TRUE

#'
"""

function simple_paths(g::Union{NamedArray,AbstractMatrix},start::String,target::String;shortest=false)
    rows,cols = size(g)
    if rows != cols error("Input g must be a square adjacency matrix when provided") end
    if isa(g, NamedArray)
        A = copy(g)
    else
        nms = autonames(rows)
        A = NamedArray(g,(nms,nms),("row","col"))
    end
    all_paths = Vector{String}[]
    function dfs(start,target,path_so_far)
        new_path = push!(copy(path_so_far),start)
        if start == target
            push!(all_paths,new_path)
            return
        end
        row_idx = findfirst(names(A, 1) .== start)
        if row_idx !== nothing
            neighbors_idx = findall(A[row_idx, :] .!= 0)
            neighbor_names = names(A, 2)[neighbors_idx]

            for next_node in neighbor_names
                if !(next_node in new_path)
                    dfs(next_node, target, new_path)
                end
            end
        end
    end
    dfs(start,target,String[])
    res = Dict{Int,Vector{String}}()
    for (i,p) in enumerate(all_paths)
        res[i] = p
    end
    if shortest && !isempty(res)
        min_len = minimum(length, values(res))
    filter!(p -> length(p.second) == min_len, res)
    end
    return res
end

"""#'
<a name="u2d"> </a>
**u2d(g)** 

> Create a directed graph out of an undirected graph.

> This function creates a directed graph from an undirected one by the given input nodes. Input nodes can be chosen by names or a number for random selection of input nodes will be given.
Input nodes will have at least shortest path distance to other input nodes of pathlength two.
Selected input nodes will draw in each iteration outgoing edges to other nodes in the nth iteration neighborhood. The input
nodes will alternatively select the next edges on the path to not visited nodes. All edges will be only visited onces.

> Arguments:

> - _g_ - an adj. matrix created with _gnew_.
  - _input_ - number or names of input nodes in the graph, if number of input nodes is smaller than number of components, for each component one input node is automatically created.
  - _negative_ proportion of inhibitive associations in the network value between 0 and 1 are acceptable, Default 0.0
  - _shuffle_ - should just the edge directions beeing shuffled, if TRUE the graph will be very random without a real structure or chains of associations, default: FALSE
 
> Returns: an adjacency matrix

> Examples:

> ```{jl label=u2d}
  G=gnew(type="angie",nodes=7,edges=9)
  deg(G,mode='in')
  U=d2u(G)
  H=u2d(U,input=c("G","E"))
  G == H
  I=u2d(U,shuffle=true)
  G == I
> ```
"""

function u2d(g::Union{NamedArray,AbstractMatrix};input=2,negative=0.0,shuffle=false)
    if g isa NamedArray
        umat = copy(g)
        nms = NamedArrays.names(g,1)
    else
        nms = autonames(size(g,1))
        umat = NamedArray(g,(nms,nms),("row","col"))
    end
    if shuffle
        idx = Random.shuffle(1:length(nms))
        h = umat[idx,idx]
        for i in axes(h,1)
            for j in i:last(axes(h,2))
                h[i,j] = 0
            end
        end
        h = h'
        return h[nms, nms]
    else
        if !issymmetric(umat)
            error("Error: Adjacency matrix must be symmetric")
        end
        if negative < 0 || negative > 1
            error("Error: negative proportions must be within 0 and 1")
        end
        # undirected matrix
        U = umat.array
        # future directed matrix
        D = zeros(eltype(U),size(U))
        node_idx = Int[]
        if input isa Number 
            comps = components(U)
            excluded_mask = falses(size(U,1))
            if maximum(comps) > 1
                for i in 1:maximum(comps)   
                    comp_nodes = findall(x -> x == i, comps)
                    if length(comp_nodes) > 1
                        node = rand(comp_nodes)
                        push!(node_idx,node)
                    end
                end
                for idx in node_idx
                    excluded_mask[idx] = true
                    nbs = findall(x -> x != 0, U[idx, :])
                    excluded_mask[nbs] .= true
                end
            end
            
            n = input - length(node_idx)
            while n > 0
                cands = findall(!,excluded_mask)
                if isempty(cands)
                    break
                end

                node = rand(cands)
                push!(node_idx,node)
                n -= 1
                excluded_mask[node] = true
                nbs = findall(x -> x != 0, U[node, :])
                excluded_mask[nbs] .= true
            end
        else
            for name in input
                idx = findfirst(==(name), nms)
                if !isnothing(idx)
                    push!(node_idx, idx)
                end
            end
        end
        queue= copy(node_idx)
        visited_edges = Set{Tuple{Int, Int}}()
        visited_nodes = Set(queue)
        while !isempty(queue)
            curr = popfirst!(queue)
            curr_nbs = findall(x -> x != 0, U[curr, :])
            for nb in curr_nbs
                if !((curr, nb) in visited_edges) && !((nb, curr) in visited_edges)
                    D[curr, nb] = 1
                    push!(visited_edges, (curr, nb))
                    if !(nb in visited_nodes)
                        push!(visited_nodes, nb)
                        push!(queue, nb)
                    end
                end
            end
        end
        if negative > 0
            edge_indices = findall(x -> x == 1, D)
            n_neg = floor(Int, length(edge_indices) * negative)
            if n_neg > 0
                neg_indices = sample(edge_indices, n_neg, replace = false)
                D[neg_indices] .= -1
            end
        end
    end
    return NamedArray(D, (nms, nms), ("row", "col"))    
end



"""#'
## Helper Functions
<a name="pmatch"> </a>
**pmatch(type::String, types::Tuple{Vararg{String}})**

> Partial match helper similar to R's pmatch.
"""

function pmatch(type::String, types::Tuple{Vararg{String}})
    matches = filter(t -> startswith(t, type), types)
    if length(matches) == 0
        error("Unknown type: $type. Options are: $(join(types, ", "))")
    elseif length(matches) > 1
        error("Ambiguous type: $type. Matches: $(join(matches, ", "))")
    end
    matches[1]
end

"""#'

<a name="getChains"> </a>
**getMiddleChain(forchain,mt2)**
"""

function getMiddleChain(forchain,mt2)
    res = String[]
    for fi in 2:(length(forchain)-1)
        fl = forchain[fi]
        found = false
        for si in (fi+1):length(forchain)
            sl = forchain[si]
            mt2 = mt2[sortperm(mt2[:,sl],rev=true),:]
            schain=reverse(NamedArrays.names(mt2,1))
            mt2 = mt2[sortperm(mt2[:,fl],rev=true),:]
            fchain=NamedArrays.names(mt2,1)
            if fchain == schain
                # collect results
                res = fchain
                found = true
                break
            end
        end
        if found
            break
        end
    end
    return(res)
end

function getChains(cor_mt::NamedArray; square=true,top=10,threshold=0.01, maxl=3)
    # remove non-correlated values
    #cor_mt[abs(cor.mt)<0.05]=0
    cormt = abs.(cor_mt)
    if square
        cormt = cormt.^2
    else
        threshold = 0.1
    end
    if top > size(cormt,1)
        top = size(cormt,1)
    end
    results = Dict{String,Any}()
    chained = []
    for i in axes(cormt,1)
        # for each node create an ordered list of nodes
        node = NamedArrays.names(cormt,2)[i]
        mt = cormt[sortperm(cormt[:,i],rev =true),:][1:top,:]
        # ignore nodes without any correlation to any other node
        # might be ignored
        # initial full chain
        chain = NamedArrays.names(mt,1)
        cormt2 = cormt[chain,chain]
        j = length(chain)
        lchain = chain[j]
        mt = cormt2[sortperm(cormt2[:,lchain],rev=true),:]
        # reverse chain
        revchain = reverse(NamedArrays.names(mt, 1))
        forchain = chain
        # as long reversed revchain and chain are different
        # shorten until the same chain is found in both
        # or it is too short
        # direct chains
        l = length(forchain)
        while l > maxl
            l = l-1
            lchain = chain[l]
            forchain = forchain[1:l]
            mt2 = cormt2[forchain,forchain]
            mt2 = mt2[sortperm(mt2[:,lchain],rev=true),:]
            if abs(mt2[size(mt2,1),lchain]) < threshold
                continue
            end
            # revchain is shorter automatically
            # as we have reduced mt2
            revchain = reverse(NamedArrays.names(mt2,1))
            if revchain == forchain
                # collect results
                key = "a-chain-"*node
                results[key] = forchain
                break
            end
            # otherwise
            # new: try all chains with node iside, not only
            # in the beginning as above
            res = getMiddleChain(forchain,mt2)
            if length(res) > 0
                key = "m-chain-"*node
                results[key] = res
                break #TODO remove break??
            end
        end
    end
    return(results)
end

function chains2edgelist(chainlist)
    edgelists = String[]
    for chain in values(chainlist)
        for i in 1:(length(chain)-1)
            edge = "$(chain[i])--$(chain[i+1])"
            push!(edgelists, edge)
        end
    end
    return(edgelists)
end

function removeNonsignifGraphEdges(A::NamedArray,p_value;alpha=0.05,kwargs...)
    if sum(A) == 0
        return(A)
    end
    for i in 1:(size(A,2)-1)
        for j in i:size(A,2)
            if p_value[i,j] > alpha
                A[i,j] = A[j,i]
            end
        end
    end
    return(A)
end

function cor_p_values(r, n)
    df = n - 2
    statistic = sqrt(df) * r / sqrt(1 - r^2)
    p = Distributions.cdf(TDist(df),statistic)
    return(2 * min(p, 1 - p))
end

function kendall_p_values(tau, n)
    if n < 3 return(1.0) end
    numerator = 3 * tau * sqrt(n * (n - 1))
    denominator = sqrt(2 * (2 * n + 5))
    z = numerator / denominator
    p =2 * Distributions.ccdf(Distributions.Normal(), abs(z))
    return(p)
end

function pmatrix(M::NamedArray;method="pearson")
    ncd = size(M,2)
    colnames = NamedArrays.names(M,2)
    P = NamedArray(zeros(Float64, ncd, ncd), (colnames, colnames), ("row", "col"))
    if method == "spearman"
        M_work = copy(M)
        for i in 1:ncd
            col = M_work[:, i]
            mask = .!ismissing.(col)
            if any(mask)
                M_work[mask, i] .= StatsBase.tiedrank(col[mask])
            end
        end
    else
        M_work = M
    end
    r_raw = StatsBase.pairwise(Statistics.cor, eachcol(M_work), skipmissing=:pairwise)
    r = NamedArray(r_raw, (colnames, colnames), ("row", "col"))
    valid_mask = Float64.(.!ismissing.(M))
    N_mat = valid_mask' * valid_mask
    for i in 1:(ncd-1)
        for j in (i+1):ncd
            N_val = N_mat[i, j]
            
            r_val = r_raw[i,j]
            p_val = cor_p_values(r_val, N_val)
            
            P[i,j] = p_val
            P[j,i] = p_val
        end
    end
    return(Dict("r"=>r, "P"=>P))
end

function custom_p_adjust(pvals::AbstractVector{T}, method::String="bonferroni") where T <: Real    n = length(pvals)
    if n <= 1
        return pvals
    end

    if method == "bonferroni"
        return min.(pvals .* n, 1.0)
    elseif method == "holm"
        idx = sortperm(pvals)
        p_sorted = pvals[idx]
        multipliers = (n .- (0:n-1))
        p_adj = p_sorted .* multipliers

        for i in 2:n
            p_adj[i] = max(p_adj[i], p_adj[i-1])
        end

        p_adj = min.(p_adj, 1.0)
        inv_idx = sortperm(idx)
        return p_adj[inv_idx]

    elseif method == "BH" || method == "fdr"
        idx = sortperm(pvals)
        p_sorted = pvals[idx]
        ranks = 1:n
        p_adj = p_sorted .* (n ./ ranks)
        
        for i in n-1:-1:1
            p_adj[i] = min(p_adj[i], p_adj[i+1])
        end

        p_adj = min.(p_adj, 1.0)
        inv_idx = sortperm(idx)
        return p_adj[inv_idx]
    else
        return pvals
    end
end

function corTest(data::DataFrame;method="pearson",p_adjust=nothing)
    mat_df = Matrix(data)
    colnames = names(data)
    named_df = NamedArray(mat_df, (string.(1:size(mat_df,1)), colnames), ("Samples", "Vars"))

    cormt = nothing
    pvalue = nothing

    if method == "pearson" || method == "spearman"
        res = pmatrix(named_df,method=method)
        cormt = res["r"]
        pvalue = res["P"]

        for i in axes(pvalue,1)
            pvalue[i,i] = 0
        end

        for i in axes(cormt,1)
            cormt[i,i] = 1
        end

    elseif method == "kendall"
        corraw = StatsBase.pairwise(Statistics.corkendall,eachcol(named_df), skipmissing=:pairwise)
        ncd = size(named_df,2)
        cormt = NamedArray(corraw,(colnames, colnames), ("row", "col"))
        pvalue = NamedArray(zeros(Float64,ncd,ncd),(colnames, colnames), ("row", "col"))
        for i in 1:(ncd-1)
            for j in (i+1):ncd
                N_val = count(k -> !ismissing(data[k,i]) && !ismissing(data[k,j]), 1:size(data, 1))
                tau_val = cormt[i,j]

                p_val = kendall_p_values(tau_val, N_val)

                pvalue[i,j] = p_val
                pvalue[j,i] = p_val
            end
        end
    end
    if !isnothing(p_adjust)
        mask = triu(trues(size(pvalue)), 1)
        raw_p = pvalue[mask]
        adj_p = custom_p_adjust(raw_p, p_adjust)
        pvalue[mask] = adj_p
        
        rows = axes(pvalue,1)
        for i in rows
            for j in (i+1):last(rows)
                pvalue[j,i] = pvalue[i,j]
            end
        end
    end
    return (Dict("r"=>cormt,"p_value"=>pvalue,"p_adjust"=>p_adjust))
end


function data2chainGraph(data::Union{DataFrame, NamedArray};method="pearson",square=true,threshold=0.01,maxl=3,top=10,
    p_adjust="none",alpha=0.01,cor_p_value=nothing)

    colnames = names(data)
    if method == "mi"
        return("no implementation yet")
    elseif method == "rpart"
        return("no implementation yet")
    elseif method == "cor.fk"
        return("no implementation yet")
    else
        if method == "pearson"
            cormt = NamedArray(Statistics.cor(Matrix(data)),(colnames,colnames),("row","col"))
        elseif method == "kendall"
            cormt = NamedArray(StatsBase.corkendall(Matrix(data)),(colnames,colnames),("row","col"))
            square = true
        elseif method == "spearman"
            cormt = NamedArray(StatsBase.corspearman(Matrix(data)),(colnames,colnames),("row","col"))
        else
            error("Unknown correlation method: $method")
        end
        test_result = corTest(data, method=method, p_adjust=p_adjust)
        cor_p_value = test_result["p_value"]
    end
        
    # if all pairs are NaN
    cor_p_value[isnan.(cormt)] .= 1.0
    replace!(cormt, NaN => 0.0)
    
    #Build adj. matrix from chains
    chains = getChains(cormt,square=square,threshold=threshold,maxl=maxl,top=top)
    edgelist = chains2edgelist(chains)
    A = NamedArray(zeros(Float64,size(cormt,1),size(cormt,1)),(colnames,colnames),("row","col"))
    for edge in edgelist
        nodes = split(edge,"--")
        A[nodes[1],nodes[2]] = 1
        A[nodes[2],nodes[1]] = 1
    end
    
    if maximum(cor_p_value) == 0
        # cor matrix was given
        # set all non edges to not significant
        # as we can't calculate p-values
        cor_p_value[A .== 0] .= 1
    end

    if alpha < 1
        # remove nons significant edges
        A = removeNonsignifGraphEdges(A,cor_p_value,alpha=alpha)
    end

    asgm = Dict("theta"=>A,
                "p_values"=>cor_p_value,
                "data"=>data,
                "sigma"=>cormt,
                "method"=>method,
                "threshold"=>threshold,
                "alpha"=>alpha,
                "chains"=>chains)
    return asgm
end

function ReduceChains(g)
    asgm = g
    ichains = []
    chs = sort(collect(keys(asgm["chains"]))) 
    for cho in chs
        if length(asgm["chains"][cho]) == 1
            continue
        end
        co = asgm["chains"][cho]
        for chi in chs
            if length(asgm["chains"][chi]) == 1
                continue
            end
            if length(asgm["chains"][cho]) == 1
                continue
            end

            if chi == cho continue end
            ci = asgm["chains"][chi]
            if length(co) > length(ci) continue end
            if length(co) == length(ci)
                if co == ci || co == reverse(ci)
                    push!(ichains,chi)
                    asgm["chains"][chi] = [""]
                end
            else
                cop = join(co)
                cipo = join(ci)
                cipr = join(reverse(ci))
                if occursin(cop,cipo)
                    push!(ichains,cho)
                    asgm["chains"][cho] = [""]
                elseif occursin(cop,cipr)
                    push!(ichains,cho)
                    asgm["chains"][cho] = [""]
                end
            end
        end
    end
    for ch in ichains
        delete!(asgm["chains"],ch)
    end
    return asgm
end

function pagerank(A::NamedArray;d=0.85,iter_max=100)
    if mode == "undirected"
        A = d2u(A)
    end
    N = size(A,2)
    v = fill(1/N,N)
    vlast = zeros(N)
    A_hat = A .* d .+ (1-d) ./ N
    for i in 1:iter_max
        v = vec(A_hat' * v)
        v .+= (1-d) / N
        v ./= sum(v)
        if maximum(abs.(v .- vlast)) < 1e-6
            break
        end
        vlast = copy(v)
    end
    nms = NamedArrays.names(A,2)
    v = NamedArray(v,(nms,), ("Node",))
    return v
end

function katzcent(g::NamedArray; alpha=0.1, norm=true)
    if alpha < 0.0 || alpha > 0.2 error("Valid alpha is between 0.0-0.2") end
    nr = size(g,2)
    # check alpha
    max_lambda = real(LinearAlgebra.eigmax(g))
    maxEv = 1.0 /max_lambda
    if alpha <= 0 || alpha >= maxEv error("Invalid alpha value.") end
    res = (I - alpha * g') \ ones(nr)
    nms = NamedArrays.names(g,2)
    res = NamedArray(res,(nms,),("Node",))
    if norm
        res .-= 1
        res = res ./ maximum(res)
    end
    return res
end

function betweenness(g::NamedArray;directed=true)
    nrow,ncol = size(g)
    nms = NamedArrays.names(g,2)
    B = NamedArray(zeros(ncol),(nms,),("Node",))
    mode_str = directed ? "directed" : "undirected"
    SP = shortest_paths(g,mode=mode_str)
    for i in 1:nrow
        start_j = directed ? 1 : (i + 1)
        for j in start_j:nrow
            if i == j continue end
            if SP[i,j] > 1 && SP[i,j] < Inf
                paths = simple_paths(g,nms[i],nms[j],shortest=true)
                num_paths = length(paths)
                if num_paths > 0
                    weight = 1.0 / num_paths
                    for p in values(paths)
                        for node_name in p[2:end-1]
                            B[node_name] += weight
                        end
                    end
                end
            end
        end
    end
    if !directed
        B .= B .* 2
    end
    return B
end

function closeness(g::NamedArray; norm=false)
    if maximum(components(g)) > 1
        res = fill(NaN, size(g, 1))
    else
        sp = shortest_paths(g)
        res = Vector{Float64}(undef, size(g, 1))
        for i in axes(sp, 1)
            res[i] = 1.0 / sum(sp[i, :])
        end
        if norm
            res = (size(g, 1) - 1) .* res
        end
    end
    nms = NamedArrays.names(g, 1)
    res = NamedArray(res, (nms,), ("closeness",))
    return res
end

function harmonic(g::NamedArray; norm=false)
    sp = shortest_paths(g)
    sp[diagind(sp)] .= Inf
    res = vec(sum(1.0 ./ sp, dims=2))
    if norm
        res ./= (size(g, 1) - 1)
    end
    nms = NamedArrays.names(g, 1)
    res = NamedArray(res, (nms,), ("harmonic",))
    return res
end


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
    println(argv)  
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
end
