
"""
    struct CVFEMSystem

Struct which contains grid, cell evaluator, bnode evaluator and usertdata.
"""
struct CVFEMSystem{G<: ExtendableGrid,FC<:Function,FB<:Function, UD}
    grid::G
    celleval::FC
    bnodeeval::FB
    userdata::UD
    num_species::Int
end


num_species(sys)=sys.num_species

"""
    zeros(sys)

Return a zero solution vector of system.
"""
Base.zeros(sys)=zeros(num_species(sys), num_nodes(sys.grid))


"""
    eval_res_jac!(sys,U,Uold,Res,Jac,time, tstep)

Evaluate global residual and jacobian for one time step of system.
"""
function eval_res_jac!(sys::CVFEMSystem{G,FC,FB,UD}, U, Uold, Res, Jac, time, tstep) where {G,FC,FB,UD}
    (;grid, celleval, bnodeeval, userdata)=sys

    coordinates = grid[Coordinates]
    cellnodes = grid[CellNodes]
    bfacenodes = grid[BFaceNodes]
    bfregions = grid[BFaceRegions]
    nspec=num_species(sys)
    
    spacedim = size(coordinates, 1)
    celldim = spacedim + 1
    celldata = CellData(coordinates, nspec, time, tstep)
    ncells = size(cellnodes, 2)
    
    bnodedata = BNodeData(coordinates)
    nbfaces = size(bfacenodes, 2)

    
    wrap_celleval(y, u) = celleval(y, u, celldata, userdata)
    wrap_bnodeeval(y, u) = bnodeeval(y, u, bnodedata, userdata)
    
    uclocal = zeros(nspec,celldim)
    yclocal = zeros(nspec,celldim)
    cresult = DiffResults.JacobianResult(yclocal, uclocal)
    cconfig = ForwardDiff.JacobianConfig(wrap_celleval, yclocal, uclocal)

    
    
    ublocal = zeros(nspec,1)
    yblocal = zeros(nspec,1)
    bresult = DiffResults.JacobianResult(yblocal, ublocal)
    bconfig = ForwardDiff.JacobianConfig(wrap_bnodeeval, yblocal, ublocal)

    Lg=LinearIndices(U)
    Lc=LinearIndices(uclocal)
    Lb=LinearIndices(ublocal)

    
    for icell = 1:ncells
        update!(celldata, coordinates, cellnodes, icell)
        # Copy solution to ulocal
        for ispec=1:nspec
            for il = 1:celldim
                ig=cellnodes[il, icell]
                @views uclocal[:,il].= U[:,ig]
                @views celldata.uold[:,il] .= Uold[:,ig]
            end
        end
        
        # AD residual and jacobian
        yclocal.=0.0
        ForwardDiff.vector_mode_jacobian!(cresult, wrap_celleval, yclocal, uclocal, cconfig)
        cres = DiffResults.value(cresult)
        cjac = DiffResults.jacobian(cresult)

        # Assemble into global data
        for il = 1:celldim
            for ispec=1:nspec
                ig = cellnodes[il, icell]
                Res[ispec,ig] += cres[ispec,il]
                for jl = 1:celldim
                    jg = cellnodes[jl, icell]
                    for jspec=1:nspec
                        v=cjac[Lc[ispec,il], Lc[jspec,jl]]
                        if !iszero(v)
                            Jac[Lg[ispec,ig],Lg[jspec,jg]] += v
                        end
                    end
                end
            end
        end
    end # for icell
    
    for ibface = 1:nbfaces
        vol = bfacevolume(coordinates, bfacenodes, ibface)
        for il = 1:spacedim
            update!(bnodedata, coordinates, bfacenodes, ibface, il, vol)
            bnodedata.region=bfregions[ibface]
            ig = bfacenodes[il, ibface]
            @views ublocal[:,1].= U[:,ig]
            yblocal.=0.0
            ForwardDiff.vector_mode_jacobian!(bresult, wrap_bnodeeval, yblocal, ublocal, bconfig)
            bres = DiffResults.value(bresult)
            bjac = DiffResults.jacobian(bresult)
            @views Res[:,ig].+= bres[:,1]
            for ispec=1:nspec
                for jspec=1:nspec
                    v= bjac[Lb[ispec,1], Lb[jspec,1]]
                    if !iszero(v)
                        Jac[Lg[ispec,ig], Lg[jspec,ig]] += v
                    end
                end
            end
        end
    end # for ibface
end

"""
    solve_step!(system, U,Uold, Res, Jac, time, tstep; kwargs...)
    
Solve one implicit Euler timestep of system, or stationary problem if tstep=Inf.
"""
function solve_step!(sys::CVFEMSystem{G,FC,FB,UD},U,Uold,Res,Jac,time,tstep;
                    tol = 1.0e-10,
                    maxiter = 20,
                    damp_initial=1.0,
                    damp_growth=1.2,
                    log=false)  where {G,FC,FB,UD};
    it = 1
    nalloc = 0
    damp=damp_initial
    while it < maxiter
        nonzeros(Jac) .= 0.0
        Res .= 0.0
        eval_res_jac!(sys,U,Uold, Res, Jac, time, tstep)
        update=Jac\vec(Res)
        vec(U).-=damp.*update
        nm = norm(update, Inf)
        if log
            @info it, nm
        end
        if nm < tol
            if nalloc > 0
                @warn "allocations in assembly loop"
            end
            return U
        end
        damp=min(damp*damp_growth,1.0)
        it += 1
    end  # while it
    error("no convergence after maxiter=$(maxiter) iterations")
end


"""
    solve(sys; kwargs...)

Solve system.
Keyword arguments and their defaults
- `times=nothing`: AbstractVector of time discretization points. If not given solve stationary problem.
- `inival=zeros(sys)`: initial value
- `tol = 1.0e-10`: stopping tolerance for Newton iteration. Compared with infinity norm of Newton update.
- `maxiter = 20` Maximum number of Newton iterations
- `damp_initial=1.0`: Initial value of damping for Newton
- `damp_growth=1.2`: Newton damp growth factor
- `log=false`: Trigger logging
"""
function SciMLBase.solve(sys::CVFEMSystem{G,FC,FB,UD};
                         times=nothing,
                         inival=zeros(sys), kwargs...) where {G,FC,FB,UD}
    (;grid)=sys
    nspec=num_species(sys)
    N = num_nodes(grid)
    ndof=N*nspec
    Jac = ExtendableSparseMatrix(ndof, ndof)
    Res = zeros(nspec,N)
    if isa(inival,Function)
        inival=map(inival,grid)
        U = zeros(nspec,N)
        U[1,:]=inival
    elseif isa(inival, Number)
        U=fill(Float64(inival),nspec,N)
    else
        U = copy(inival)
    end
    Uold=copy(U)
    if isnothing(times)
        solve_step!(sys,U,Uold,Res,Jac,0.0,Inf; kwargs...)
    else
        tsol=TransientSolution([copy(Uold)],[times[1]])
        for itime=2:length(times)
            tstep=times[itime]-times[itime-1]
            Uold.=U
            solve_step!(sys,U,Uold,Res,Jac,times[itime],tstep; kwargs...)
            append!(tsol, times[itime], copy(U))
        end
        tsol
    end
end


