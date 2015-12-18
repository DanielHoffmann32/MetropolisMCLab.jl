module MetropolisMCLab

# package code goes here
export Vharmonic,
DX,
OneDimMetropolisMC

# Metropolis MC code -----------------------------

#additional packages that we need
using StatsBase
using Distributions
using HypothesisTests

#harmonic potential
Vharmonic(x::Float64, a::Float64=1., b::Float64=0.) = a*(x-b)^2

#a simple move function
DX(dxmax::Float64) = dxmax * 2. * (rand()-0.5)

#a move function for a periodic system on [0,2π[
xtry_p(xold::Float64, dxmax::Float64) = 
    mod2pi(xold + dxmax * 2. * (rand()-0.5))

#-------------------------------------------------------------

#         1D-Metropolis Monte Carlo (MC) function 

#-------------------------------------------------------------

function 
    OneDimMetropolisMC(
        X0::Float64, 
        Nsteps::Int64, 
        E::Function, 
        RT::Float64, 
        DXmax::Float64, 
        Epars...
    )
    
    #initialize position and energy
    X = zeros(Nsteps)
    X[1] = X0
    Ex = E(X[1],Epars...)

    #generate Markov chain
    for i in 1:(Nsteps-1)
        Xtry = X[i] + DX(DXmax)
        Etry = E(Xtry,Epars...)

        #apply Metropolis criterion
        if (Etry < Ex || rand() < exp(-(Etry-Ex)/RT))
            X[i+1] = Xtry
            Ex = Etry
        else
            X[i+1] = X[i]
        end
    end

    #result: array of sampled positions
    return X 

end

end # module
