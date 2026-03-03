N = 8 # number of qubits
nsamp = parse(Int, ARGS[1]) # number of shots
seed = parse(Int, ARGS[2]) # random seed
tau = 1.0
gamma = 1.0

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



#0から1.6までを40等分したリスト
h1_list = collect(LinRange(0.0, 1.6, 40))
h2_list = [-1.35, -1.285, -1.225, -1.154, -1.109, -1.079,-1.049, -1.024, -1.0009, -1.004, -0.3531, -0.2479,-0.1377, -0.02755, 0.0, 0.09766, 0.2229, 0.3631, 0.5033, 0.6636, 0.8439]
ndata = length(h1_list)*length(h2_list)
train_cs = []

#CSデータ読み込み
#順番注意
for (ih2, h2) in enumerate(h2_list)
    for (ih1, h1) in enumerate(h1_list)
        #println("Loading data for h1=$(h1), h2=$(h2)")
        x = load("data_T=$(nsamp)/cs_nsamp=$(nsamp)_h1=$(ih1)_h2=$(ih2)_seed=$(seed).jld2","cs")
        y = to_ClassicalShadow_SK(x, tau, gamma)
        push!(train_cs, y)
    end
end

#Gram matrix computation
G = zeros(Float64, ndata, ndata)
for i=1:ndata
    #println("Computing row $(i) / $(ndata)")
    for j=i:ndata
        cs1 = train_cs[i]
        cs2 = train_cs[j]
        G[i,j] = G[j,i] = shadow_kernel(cs1, cs2)
    end
end
filename = @sprintf("gram_matrix_T=%d/SK_nsamp=%d_seed=%d.csv", nsamp, nsamp, seed)
pynp.savetxt(filename, G, delimiter=",")


#Label data saving
#順番注意
label_list = []
for (ih2, h2) in enumerate(h2_list)
    for (ih1, h1) in enumerate(h1_list)
        label = load("data_T=$(nsamp)/cs_nsamp=$(nsamp)_h1=$(ih1)_h2=$(ih2).jld2","label")
        push!(label_list, label)
    end
end
filename = @sprintf("gram_matrix_T=%d/label.csv", nsamp)
pynp.savetxt(filename, label_list, delimiter=",")