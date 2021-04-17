"""Quantum Heisenberg Hamiltonian on a one-dimensional chain
 H = Σ_i J (vec(S)_i ⊗ vec(S)_i+1) + Σ_i h_i S_i^ax
 where vec(S) = (S^x, S^y, S^z),
 S^x,y,z = 1/2 σ^x,y,z the Pauli matrix
 and o the tensor product. 
"""
module Hamiltonian
export Heisenberg1DHamiltonian
using LinearAlgebra



function Pauli(ax::String)::Array
    """
    Returns Pauli matrix of given axis. Note the factor of 1/2.
    """
    if ax == "x"
        return 1/2 * Array{Float64,2}([0 1.; 1 0])
    elseif ax == "y"
        return 1/2 * Array{ComplexF64,2}([0 -1im; 1im 0])
    elseif ax == "z"
        return 1/2 * Array{Float64,2}([1 0; 0 -1])
    else
        error("Axis $(ax) is not implemented. Use axis = \"x\", \"y\", or \"z\" instead!")
    end
end


function SpinProduct(ax::String)::Array
    """
    Product of two Pauli-matrices along axis ax.
    σ_{ax} ⊗ σ_{ax}
    Parameters:
        ax  ::String    Axis [x,y,z]
    """
    spin = Pauli(ax)
    return kron(spin,spin)
end

function IdKron(n::Int64)::Array
    """
    Recursive Kronecker-product of n identities. 
    1⊗1⊗...⊗1 (n-times)
    Parameters:
        n   ::Inte64    Number of Kronecker-products
    """
    id = [1 0; 0 1]
    if n == 1
        return id
    else
        return kron(id,IdKron(n-1))
    end
end

function HeisenbergGenerator(nfirst::Int64,nlast::Int64,size::Int64,J::Array)::Array
    """
    Helper function to generate series of tensor products for 1d Heisenberg Hamiltonian.
    Parameters:
        nfirst  ::Int64          Number of operators in front of Si ⊗ Sj
        nlast   ::Int64          Number of operators after Si ⊗ Sj
        size    ::Int64          Chain length
        J       ::Array<Float64> Array of interaction strengths [Jx, Jy, Jz]
    """
    axes = ["y","z"]
    if nfirst == 0
        s = J[1] * Pauli("x")
        HG = kron(kron(s,s),IdKron(size-2))
        for i in 1:length(axes)
            s = J[i+1] * Pauli(axes[i])
            HG += kron(kron(s,s),IdKron(size-2))
        end
    elseif nlast == 0
        s =J[1] * Pauli("x")
        HG = kron(kron(IdKron(size-2),s),s)
        for i in 1:length(axes)
            s = J[i+1] *Pauli(axes[i])
            HG += kron(kron(IdKron(size-2),s),s)
        end
    else
        s = J[1] * Pauli("x")
        HG = kron(kron(kron(IdKron(nfirst),s),s),IdKron(nlast))
        for i in 1:length(axes)
            s = J[i+1] * Pauli(axes[i])
            HG += kron(kron(kron(IdKron(nfirst),s),s),IdKron(nlast))
        end
    end
    return HG
end

function HeisenbergOnsite(ax::String,size::Int64,hfields::Array)::Array
    """
    On-Site term of Heisenberg model: Σ_i h_i S_i^ax
    Parameters:
        ax          ::String         Axis of external field {"x","y","z"}
        size        ::Int64          Chain length 
        hfields     ::Array<Float64> Array of field strengths 
        !NOTE!      len(hfields)    == size 
    
    Returns:
        Hh          ::Array<Float64>    On-Site Hamiltonian Matrix.
    """
    s = Pauli(ax)
    Hh = hfields[1]*kron(s,IdKron(size-1))
    for i in 2:size-1
        Hh += hfields[i]*kron(kron(IdKron(i-1),s),IdKron(size-i))
    end
    Hh  += hfields[end]*kron(IdKron(size-1),s)
    return Hh
end



function Heisenberg1DHamiltonian(size::Int64, J::Array, hfields::Array, hax::String, bc::String)::Array
    """
    Generates the Heisenberg Hamiltonian for a one-dimensional chain with nearest-neighbor interaction.
    Parameters:
         size       ::Int64                 Chain length
         J          ::Array<Float64>        Array of interaction strengths [Jx, Jy, Jz]
         hfields    ::Array<Float64>        Array of field strengths 
         !NOTE!      len(hfields)    == size 
         hax        ::String                Axis of external field {"x","y","z"}
         bc         ::String                Boundary condition {"pbc","obc"} 
    Returns:
        Hamiltonian ::Array<ComplexF64>     Hamiltonian-Matrix.
    """
    
    axes = ["x","y","z"]

    Hh   = HeisenbergOnsite(hax,size,hfields)
    H    = HeisenbergGenerator(0,size-2,size,J)
    # Periodic BC
    if bc == "pbc"
        for i in 1:length(axes)
            s = J[i] * Pauli(axes[i])    
            H += kron(kron(s,IdKron(size-2)),s)
        end
    end

    for i in 2:(size-1)
        H += HeisenbergGenerator(i-1,size-1-i,size,J)
    end
    return H + Hh
end

end
