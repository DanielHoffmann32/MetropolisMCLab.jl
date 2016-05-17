module MetropolisMCLab

# package code goes here
export block_average,
Vharmonic, Vharmonic2D,
Vflat2D, Vcos2D, Vmultiminima2D,
xtry, xtry_p, rtry_p, rtry,
OneDimMetropolisMC,
TwoDimMetropolisMC,
expansion_clustering,
dists2D


# Metropolis MC code -----------------------------

#additional packages that we need
using StatsBase
using Distributions, DataFrames
using HypothesisTests

#harmonic potential in 1D
Vharmonic(x::Float64, a::Float64=1., b::Float64=0.) = a*(x-b)^2

#harmonic potential in 2D
Vharmonic2D(r::Array{Float64,1}, a::Float64=1., b::Float64=0., c::Float64=0.) =
    a*((r[1]-b)^2+(r[2]-c)^2)

#flat potential in 2D
Vflat2D(r::Array{Float64,1}) = 0.

#cosine multiwell potential in 2D
Vcos2D(r::Array{Float64,1}) = cos(r[1])*cos(r[2])

#multiminima potential in 2D
Vmultiminima2D(r::Array{Float64,1}, a::Float64=0.02, b::Float64=0., c::Float64=0.) =
    a*((r[1]-b)^2+(r[2]-c)^2) + cos(r[1])*cos(r[2])

#a simple move function in 1D
xtry(xold::Float64, dxmax::Float64) = 
    xold + dxmax * 2. * (rand()-0.5)

# a move function for a periodic system on [0,2pi[
xtry_p(xold::Float64, dxmax::Float64) = 
    mod2pi(xold + dxmax * 2. * (rand()-0.5))

# a move function for a periodic system on [0,2pi[ x [0,2pi[
rtry_p(rold::Array{Float64,1}, drmax::Float64) = 
    [mod2pi(rold[1] + drmax * 2. * (rand()-0.5));
     mod2pi(rold[2] + drmax * 2. * (rand()-0.5))]

# a move function for 2D system
rtry(rold::Array{Float64,1}, drmax::Float64) = 
    [rold[1] + drmax * 2. * (rand()-0.5); 
     rold[2] + drmax * 2. * (rand()-0.5)]

#compute distances between ref-point (x_ref, y_ref) and points in array (x, y)
function dists2D(x_ref::Float64, y_ref::Float64, x::Array{Float64,1}, y::Array{Float64,1})
    N = length(x)
    d = zeros(N)
    for i in 1:N
        d[i] = sqrt((x_ref-x[i])^2 + (y_ref-y[i])^2)
    end
    return d
end

#Iteratively compute distance based clusters (distance threshold "r"),
#starting from point in the trajectory "coors" that is
#closest to external reference point "ext". 
function expansion_clustering(coors::Array{Float64,2}, ext::Array{Float64,1}, r::Float64)
    #the trajectory
    N = length(coors[:,1])
    traj = 
    DataFrame(
        x = coors[:,1], 
        y = coors[:,2], 
        d = dists2D(ext[1], ext[2], coors[:,1], coors[:,2]), #init. with dists. to external reference point
        clust = zeros(Int64,N) #cluster index of each frame
    )

    #sort according to distance
    sort!(traj, cols = :d)

    n_clust = 1
    traj_assigned = DataFrame(x=[], y=[], d=[], clust=[])

    #clustering in this loop
    while true

        #we sort coordinates according to distance to reference
        sort!(traj, cols = :d)

        #all coordinates with distances to reference < r are in the current cluster
        curr_clust = (traj[:d] .<= r)

        #if there is a point in the current cluster ...
        if curr_clust[1] == true
            traj[curr_clust, :clust] = n_clust #assign frames to this cluster
            n_clust += 1
            traj_assigned = vcat(traj_assigned, traj[curr_clust,:]) #collect assigned frames
        end

        #if there are still points that have not been clustered ...
        if curr_clust[end] == false

            #collect these remaining points and use the first frame as new reference
            # (= unassigned frame with lowest distance to previous reference)
            traj = traj[!curr_clust,:]

            #recompute distances based on this reference
            traj[:d] = dists2D(traj[1,:x], traj[1,:y], collect(traj[:x]), collect(traj[:y]))
        else
            break
        end
    end
    return traj_assigned
end

"""
Splits a y-trajectory of length N into M consecutive blocks of size n. n takes values 1, 1+dn, 1+2*dn, ... 
For each block size: computes averages <y> on each block and then standard error of the mean from BSE = sigma(<y>)/\sqrt{M}.

Input:

  - y: quantity computed along a trajectory

  - Mmin: minimum number of blocks

  - dn Increment of block size

Output:

  - BSE data frame, with columns n (block sizes) and BSE (block average standard errors)
  
"""
function block_average(y::Array{Float64,1};
                       Mmin::Int64=10,
                       dn::Int64=100)
    N = length(y)
    ns = 1:dn:(N/Mmin)
    BSE = DataFrame(n = collect(ns), bse = zeros(length(ns)))
    j = 1
    for n in ns
        M = N/n
        BSE[j,:n] = n
        BSE[j,:bse] = std(map(i -> mean(y[i:(i+n-1)]), 1:n:(N-n)))/sqrt(M)
        j += 1
    end
    BSE #block average standard error as function of block size n
end

#-------------------------------------------------------------

#         1D-Metropolis Monte Carlo (MC) function 

#-------------------------------------------------------------

function 
    OneDimMetropolisMC(
    X0::Float64, 
    Nsteps::Int64, 
    U::Function, 
    RT::Float64, 
    DX::Function,
    DXmax::Float64, 
    Upars...
    )
    
    #initialize position and energy
    X = zeros(Nsteps)
    X[1] = X0
    Ux = U(X[1],Upars...)

    #generate Markov chain
    for i = 1:(Nsteps-1)
        Xtry = DX(X[i], DXmax)
        Utry = U(Xtry,Upars...)

        #apply Metropolis criterion
        if (Utry < Ux || rand() < exp(-(Utry-Ux)/RT))
            X[i+1] = Xtry
            Ux = Utry
        else
            X[i+1] = X[i]
        end
    end

    #result: array of sampled positions
    return X 
    
end

#-------------------------------------------------------------

#         2D-Metropolis Monte Carlo (MC) function 

#-------------------------------------------------------------

function 
    TwoDimMetropolisMC(
    r0::Array{Float64,1}, 
    Nsteps::Int64, 
    U::Function, 
    RT::Float64, 
    Dr::Function,
    Drmax::Float64, 
    Upars...
    )
    
    #initialize position and energy
    r = zeros(Nsteps,2)
    r[1,:] = r0
    Ur = U(r0, Upars...)

    #generate Markov chain
    for i = 1:(Nsteps-1)
        rtry = Dr(vec(r[i,:]),Drmax)
        Utry = U(rtry,Upars...)

        #apply Metropolis criterion
        if (Utry < Ur || rand() < exp(-(Utry-Ur)/RT))
            r[i+1,:] = rtry
            Ur = Utry
        else
            r[i+1,:] = r[i,:]
        end
    end

    #result: array of sampled positions
    return r
    
end

end # module
