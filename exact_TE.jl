module ExactTE
export ETimeEvo#, SuddenQuench
include("helper.jl")
using .Helper
using LinearAlgebra

function ETimeEvo(psi0,Ham,size,tend,dt,tsave,tflush,ax,filname)
    D, U = Diagonal{Float64}(eigvals(Ham)),reshape(eigvecs(Ham),2^size,2^size)
    Ud = Helper.Dagger(U)
    psi_p = psi0
    psi_c = psi0
    psi = [psi0]*(1.0+0.0*1im)
    time = [0.0]
    k = 0
    for i in 1:Int(tend/dt)
        psi_c = Ud * exp(-1im*D*i*dt) * U * psi_p
        psi_p = psi_c
        if i % tsave == 0
            push!(psi,psi_c)
            push!(time,i*dt)
        end
        if i % tflush == 0
            Helper.WriteLocSpin(time,psi,k,filname,ax,size)
            Helper.WriteSurvProb(time,k,filname,psi0,psi)
            psi = []*(1.0+0.0*1im)
            time = []
            k+=1
        end
    end
    return time,psi
end

#function SuddenQuench(psi0,Ham0,Ham1,size,tend1,tend2,tsave,dt)
#    psi1 = ETimeEvo(psi0,Ham0,size,tend1,dt,tsave)[end]
#    psi2 = ETimeEvo(psi1,Ham1,size,tend2,dt,tsave)
#    return psi2
#end
end