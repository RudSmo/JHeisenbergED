module Input
include("helper.jl")
include("exact_TE.jl")
include("hamiltonian.jl")
using .ExactTE
using .Hamiltonian
using .Helper

# Parameters
bc = "obc"
L = 6
J = [1.,1.,1.]
hax = "z"
ax = "z"
W = 0.5
h = W*rand(L)
Ham = Hamiltonian.Heisenberg1DHamiltonian(L,J,h,hax,bc)

u_state = ["u" for i in 1:Int(L/2)]
d_state = ["d" for i in 1:Int(L/2)]
state = collect(Iterators.flatten([u_state,d_state]))
psi0 = Helper.ConstructState(state)*(1.0+1im*0.0)

tend = 100.0
dt = 0.0001
tsave = 1000
tflush = 5000
dir = "data/test"
filname = "$(dir)/Random_L$(L)"
mkpath(dir)
time,psi = ExactTE.ETimeEvo(psi0,Ham,L,tend,dt,tsave,tflush,ax,filname)

end