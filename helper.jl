module Helper

include("hamiltonian.jl")
using .Hamiltonian
using LinearAlgebra
export Heisenberg1DSpectrum, SpinOperator, Dagger, Measure, AverageSpin, LocalSpin, TLocalSpin, WriteLocSpin, SurvivalProbability, WriteSurvProb

function Heisenberg1DSpectrum(size::Int64, J::Array, h::Array, hax::String, bc::String)::Tuple
    H = Hamiltonian.Heisenberg1DHamiltonian(size,J,h,hax,bc)
    return eigvals(H),reshape(eigvecs(H),2^size,2^size)
end

function SpinOperator(ax::String, i::Int64,size::Int64)::Array
    s = Hamiltonian.Pauli(ax)
    if i == 1
        return kron(s,Hamiltonian.IdKron(size-1))
    elseif i == size
        return kron(Hamiltonian.IdKron(size-1),s)
    else
        return kron(kron(Hamiltonian.IdKron(i-1),s),Hamiltonian.IdKron(size-i))
    end
end

function Dagger(vec::Array)::Array
    return transpose(conj(vec))
end

function Measure(operator::Array, state::Array)
    return real(Dagger(state) * operator * state)
end

function LocalSpin(state::Array,ax::String,size::Int64)::Array
    return [Measure(SpinOperator(ax,i,size),state)[1] for i in 1:size]
end

function TLocalSpin(state::Array,ax::String,size::Int64)::Array
    sloc = []
    for i in 1:length(state)
        loc = LocalSpin(state,ax,size)
        push!(sloc,loc)
    end
    return sloc
end

function AverageSpin(ax::String, state::Array, size::Int64)::Float64
    return 1/size * real(sum(LocalSpin(state,ax,size)))
end

function SurvivalProbability(psi0::Array,psi::Array)::Array
    PS = []
    for i in 1:length(psi)
        ps = abs(real.(Dagger(psi0)*psi[i])[1])#^2
        push!(PS,ps)
    end
    return PS
end

function WriteSurvProb(time::Array,k::Int64,filname::String,psi0::Array,psi::Array)
    if psi != []
        if k == 0
            fil = open("$(filname)_SP.dat","w")
        else
            fil = open("$(filname)_SP.dat","a")
        end
        PS = SurvivalProbability(psi0,psi)
        for i in 1:length(PS)
            write(fil,"$(time[i]) $(PS[i])\n")
        end
        close(fil)
    end
end

function State(s::String)::Array
    if s == "u"
        return [1, 0]
    else
        return [0, 1]
    end
end

function ConstructState(state::Array)::Array
    psi = State(state[1])
    for i in 2:length(state)
        psi = kron(psi,State(state[i]))
    end
    return psi
end

function WriteLocSpin(time::Array,psi::Array,k::Int64,filname::String,ax::String,size::Int64)
    if k == 0
        fil = open("$(filname)_Sloc.dat","w")
    else
        fil = open("$(filname)_Sloc.dat","a")
    end
    for i in 1:length(psi)
        sloc = Helper.LocalSpin(psi[i],ax,size)
        write(fil,"$(time[i]) ")
        for j in 1:size
            if j != size
                write(fil,"$(sloc[j]) ")
            end
            if j == size
                write(fil,"$(sloc[j])\n")
            end
        end
    end
    close(fil)
end

end