mutable struct ClassicalShadow
    paulis::Array{Int64,2}
    vals::Array{Int64,2}
    nqubits::Int64
    num_samp::Int64
end

# constructor
function ClassicalShadow(; paulis::Array{Int64,2}, vals::Array{Int64,2}, nqubits::Int64, num_samp::Int64)
    return ClassicalShadow(paulis, vals, nqubits, num_samp)
end

# +
#Sdag
function ITensors.op(::OpName"Sdag", ::SiteType"S=1/2", s1::Index)
  mat = [1 0 
         0 -im]
  return itensor(mat, s1', s1)
end
#"""#

# Obtain classical shadow from MPS
function random_measurement(nqubits::Int64, num_samp::Int64, psi::MPS)
    
    # site
    sites = siteinds(psi)
    
    # random basis
    paulis = zeros(Int64, num_samp, nqubits)
    vals = zeros(Int64, num_samp, nqubits)
    
    for k=1:num_samp
        os = []
        for i=1:nqubits
            r = rand(1:3)
            if r==1
                append!(os, [("H", i)])
                paulis[k,i] = 1
            elseif r==2
                #append!(os, [("H", i),("Sdag", i)])
                append!(os, [("Sdag", i),("H", i)])
                paulis[k,i] = 2
            elseif r==3
                append!(os, [("I", i)])
                paulis[k,i] = 3
            end
        end
        
        # basis transformation gates
        gates = ITensors.ops(os, sites)

        # sampling
        psi_trans = ITensors.apply(gates, psi; cutoff = 1e-15)
        psi_trans = ITensors.orthogonalize(psi_trans, 1)
        vals[k,:] = ITensors.sample(psi_trans) #1 corresponds to the eigenvalue of +1, and 2 corresponds to the eigenvalue of -1
        
    end
            
    cs = ClassicalShadow(; paulis, vals, nqubits, num_samp)
    
    return cs
end
# -

# Calculate Pauli expectation value. obs is an array of n components. Each component is I=0,X=1,Y=2,Z=3.
function Pauli_epval(cs::ClassicalShadow, obs::Array{Int64,1})
    paulis = cs.paulis
    vals = cs.vals
    nqubits = cs.nqubits
    num_samp = cs.num_samp
    
    # Filter qubits that is not associated with obs to 0 
    filter = zeros(Int64, length(obs))
    weight = 0
    for i=1:length(obs)
        if obs[i] == 0
            filter[i] = 0
        else
            filter[i] = 1
            weight += 1
        end
    end
    
    epval = 0.0
    for i=1:num_samp
        if filter.*paulis[i,:] == obs
            sum_vals = sum(filter.* (vals[i,:]-ones(Int64, nqubits))) # Add up all the measurement results from qubits related to obs and check the parity of the result. vals is off by 1, so restore it.
            epval +=  ((-1)^sum_vals)*(3^weight)
        end
    end
    
    return epval/num_samp     
end



########################################################################################################################################################################################################################################################################################################################################################################################################################################################################
mutable struct ClassicalShadow_SK
    paulis::Array{Int64,2}
    vals::Array{Int64,2}
    nqubits::Int64
    num_samp::Int64
    CS_x::Array{Int64,2}
    CS_y::Array{Int64,2}
    CS_z::Array{Int64,2}
    tau::Float64
    gamma::Float64
    norm::Float64
end

# constructor
function ClassicalShadow_SK(; 
            paulis::Array{Int64,2},
            vals::Array{Int64,2},
            nqubits::Int64,
            num_samp::Int64,
            CS_x::Array{Int64,2},
            CS_y::Array{Int64,2},
            CS_z::Array{Int64,2},
            tau::Float64, 
            gamma::Float64, 
            norm::Float64)
    return ClassicalShadow_SK(paulis, vals, nqubits, num_samp, CS_x, CS_y, CS_z, tau, gamma, norm)
end
# -

function to_ClassicalShadow_SK(cs::ClassicalShadow, tau::Float64, gamma::Float64)
    
    paulis = cs.paulis
    vals = cs.vals
    nqubits = cs.nqubits
    num_samp = cs.num_samp
    
    ############################################################################
    ### simple representation
    ############################################################################
    x_list = zeros(Int64, num_samp, nqubits)
    y_list = zeros(Int64, num_samp, nqubits)
    z_list = zeros(Int64, num_samp, nqubits)

    for k in 1:num_samp
        for i in 1:nqubits
            if paulis[k,i] == 1
                x_list[k,i] = 1
                y_list[k,i] = 0
                z_list[k,i] = 0
            elseif paulis[k,i] == 2
                x_list[k,i] = 0
                y_list[k,i] = 1
                z_list[k,i] = 0
            else
                x_list[k,i] = 0
                y_list[k,i] = 0
                z_list[k,i] = 1
            end
        end
    end

    # Add sign depending on measurement outcome
    sign::Array{Int64,2} = (-1) .^ vals

    CS_x::Array{Int64,2} = sign .* x_list
    CS_y::Array{Int64,2} = sign .* y_list
    CS_z::Array{Int64,2} = sign .* z_list
        
    #
    norm = 1.0
     
    ############################################################################
    ### to shadow kernel representation
    ############################################################################
    cs_sk = ClassicalShadow_SK(; paulis, vals, nqubits, num_samp, CS_x, CS_y, CS_z, tau, gamma, norm)
    
    return cs_sk
end

function shadow_kernel(cs1::ClassicalShadow_SK, cs2::ClassicalShadow_SK)
    
    nqubits = cs1.nqubits
    num_samp = cs1.num_samp
    tau = cs1.tau
    gamma = cs1.gamma
    
    cs1_x = cs1.CS_x
    cs1_y = cs1.CS_y
    cs1_z = cs1.CS_z
    cs2_x = cs2.CS_x
    cs2_y = cs2.CS_y
    cs2_z = cs2.CS_z
    
    mat::Array{Float64,2} = (4.5 * (cs1_x * cs2_x' + cs1_y * cs2_y' + cs1_z * cs2_z')) .+ (0.5 * nqubits)
    exps = sum(exp.(mat * gamma / nqubits))
    return exp(exps * tau / (num_samp^2)) / (cs1.norm * cs2.norm)
end

        
########################################################################################################################################################################################################################################################################################################################################################################################################################################################################
mutable struct ClassicalShadow_TESK
    paulis::Array{Int64,2}
    vals::Array{Int64,2}
    nqubits::Int64
    num_samp::Int64
    gamma::Float64
    tau::Float64
    exponent::Int64
    AGL::Vector{Vector{Int64}}
    feature_vec::Vector{Vector{Float64}}
    norm::Float64
end
        
        
# constructor
function ClassicalShadow_TESK(; 
            paulis::Array{Int64,2},
            vals::Array{Int64,2},
            nqubits::Int64,
            num_samp::Int64,
            gamma::Float64,
            tau::Float64,
            exponent::Int64,
            AGL::Vector{Vector{Int64}},
            feature_vec::Vector{Vector{Float64}},
            norm::Float64)
    return ClassicalShadow_TESK(paulis, vals, nqubits, num_samp, gamma, tau, exponent, AGL, feature_vec, norm)
end


function to_ClassicalShadow_TESK(cs::ClassicalShadow, gamma::Float64, tau::Float64, exponent::Int64, AGL::Vector{Vector{Int64}})

    paulis = cs.paulis
    vals = cs.vals
    nqubits = cs.nqubits
    num_samp = cs.num_samp

    ############################################################################
    ### feature vector
    ############################################################################
    feature_vec = Vector{Vector{Float64}}()
    for A in AGL
        
        #
        theta = (tau/num_samp^2)^(1.0/length(A))
        
        xi = sqrt( theta*(9.0*gamma)/(2.0*length(A)) )
        r = sqrt( theta * (1.0 + gamma/(2.0*length(A))) )

        vec = zeros(4^length(A))
        for k=1:num_samp
            vecA = [1.0]
            
            for i in A
                vec_temp = zeros(Float64, 4)
                if paulis[k,i]==1 && vals[k,i]==1 #Note that vals = 1, 2
                    vec_temp = [+xi, 0.0, 0.0, r]
                elseif paulis[k,i]==1 && vals[k,i]==2
                    vec_temp = [-xi, 0.0, 0.0, r]
                elseif paulis[k,i]==2 && vals[k,i]==1
                    vec_temp = [0.0, +xi, 0.0, r]
                elseif paulis[k,i]==2 && vals[k,i]==2
                    vec_temp = [0.0, -xi, 0.0, r]
                elseif paulis[k,i]==3 && vals[k,i]==1
                    vec_temp = [0.0, 0.0, +xi, r]
                elseif paulis[k,i]==3 && vals[k,i]==2
                    vec_temp = [0.0, 0.0, -xi, r]
                end

                vecA = kron(vecA,vec_temp)
            end
            vec += vecA
        end

        push!(feature_vec, vec)
    end
    
    #
    norm = 1.0
    
    ############################################################################
    ### to shadow kernel representation
    ############################################################################
    cs_tesk = ClassicalShadow_TESK(; paulis, vals, nqubits, num_samp, gamma, tau, exponent, AGL, feature_vec, norm)
    return cs_tesk
end

        
        
function TESK(cs1::ClassicalShadow_TESK, cs2::ClassicalShadow_TESK)   
            
    AGL = cs1.AGL
    
    kernel = 0.0
    for i=1:length(AGL)
        fv1 = cs1.feature_vec[i]
        fv2 = cs2.feature_vec[i]
        kernel += exp(dot(fv1, fv2))
    end
            
    kernel = kernel/length(AGL)
    kernel = kernel^exponent
            
    return kernel/(cs1.norm * cs2.norm)
end
