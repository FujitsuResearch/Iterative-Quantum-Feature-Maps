N = 8 # number of qubits
nsamp = parse(Int, ARGS[1]) # number of shots
seed = parse(Int, ARGS[2]) # random seed

using SparseArrays
using LinearAlgebra
using KrylovKit
using Arpack
using Plots
using CSV
using DataFrames
using Printf
using PyCall
using ITensors, ITensorMPS
using ProgressBars
using Random
using JLD2
const pynp = pyimport("numpy")

BLAS.set_num_threads(1) 
ITensors.blas_get_num_threads() 

# original functions for shadow
include("my_cs_tools_v250625.jl")

# fuctions
function ITensors.op(::OpName"Sdag", ::SiteType"S=1/2", s1::Index)
  mat = [1 0 
         0 -im]
  return itensor(mat, s1', s1)
end
#"""#
                        
function haar_random(dim::Int)    
                            
    #Random.seed!(123456789+seed)
                            
    a=randn(dim,dim)
    b=randn(dim,dim)
    c = a+b*im
                            
    # QR decomposition
    QRFactorization = qr(c)

    # Q-matrix
    Q = QRFactorization.Q

    # R-matrix
    R = QRFactorization.R

    # Lambda-matrix
    Lambda = zeros(ComplexF64, dim, dim)
    for i=1:dim
        Lambda[i,i] = R[i,i]/abs(R[i,i])
    end

    # haar random
    U = Q*Lambda
    return U            
end

# DMRG
function ZXZ_dmrg(h1, h2, N)
    sites = siteinds("S=1/2",N)

    ampo = OpSum()
    for j=1:N
        ampo += -1.0,"Z",j,"X",j%N+1,"Z",(j+1)%N+1
        ampo += -h1,"X",j
        ampo += -h2,"X",j,"X",j%N+1
    end
    H = MPO(ampo,sites)

    psi0 = randomMPS(sites,10)

    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 10,20,100,100,200)
    setcutoff!(sweeps, 1E-10)

    energy, psi = dmrg(H,psi0, sweeps)
    
    return energy, psi, sites
end



#0から1.6までを40等分したリスト
h1_list = collect(LinRange(0.0, 1.6, 40))
h2_list = [-1.35, -1.285, -1.225, -1.154, -1.109, -1.079,-1.049, -1.024, -1.0009, -1.004, -0.3531, -0.2479,-0.1377, -0.02755, 0.0, 0.09766, 0.2229, 0.3631, 0.5033, 0.6636, 0.8439]

function determine_label(h1, h2)
    label = -1  # 初期値

    # anti ferromagnetic phase
    if abs(h2 - (-1.004)) < 1e-10
        if h1>0.1
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.0009)) < 1e-10
        if h1>0.2556
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.024)) < 1e-10
        if h1>0.4111
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.049)) < 1e-10
        if h1>0.5667
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.079)) < 1e-10
        if h1>0.7222
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.109)) < 1e-10
        if h1>0.8778
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.154)) < 1e-10
        if h1>1.0333
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.225)) < 1e-10
        if h1>1.1889
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.285)) < 1e-10
        if h1>1.3444
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-1.35)) < 1e-10
        if h1>1.5000
            label = 0
        else
            label = 1
        end
    end

    # para magnetic phase
    if abs(h2 - 0.8439) < 1e-10
        if h1<0.1000
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - 0.6636) < 1e-10
        if h1<0.2556
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - 0.5033) < 1e-10
        if h1<0.4111
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - 0.3631) < 1e-10
        if h1<0.5667
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - 0.2229) < 1e-10
        if h1<0.7222
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - 0.09766) < 1e-10
        if h1<0.8778
            label = 0
        else            
            label = 1
        end
    elseif abs(h2 - 0.0) < 1e-10
        if h1<1.0
            label = 0
        else            
            label = 1
        end
    elseif abs(h2 - (-0.02755)) < 1e-10
        if h1<1.0333
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-0.1377)) < 1e-10
        if h1<1.1889
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-0.2479)) < 1e-10
        if h1<1.3444
            label = 0
        else
            label = 1
        end
    elseif abs(h2 - (-0.3531)) < 1e-10
        if h1<1.5000
            label = 0
        else
            label = 1
        end
    end

    if label == -1
        error("Label could not be determined for h1=$(h1), h2=$(h2)")
    end

    return label
end


# generate data
Random.seed!(1000*seed)
for (ih1, h1) in enumerate(h1_list)
    for (ih2, h2) in enumerate(h2_list)
        #println("Creating data for h1=$(h1), h2=$(h2)")
        energy, psi, sites = ZXZ_dmrg(h1, h2, N)
        cs = random_measurement(N,nsamp,psi)
        label = determine_label(h1, h2)
        save( "data_T=$(nsamp)/cs_nsamp=$(nsamp)_h1=$(ih1)_h2=$(ih2)_seed=$(seed).jld2", Dict("cs"=>cs, "label"=>label) )
    end
end