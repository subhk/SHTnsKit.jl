module SHTnsKitParallelCUDAExt

import SHTnsKit
import MPI
import PencilArrays
import PencilFFTs
import CUDA
import GPUArrays
import GPUArraysCore
import KernelAbstractions
import LinearAlgebra

const CUDAExt = Base.get_extension(SHTnsKit, :SHTnsKitGPUExt)
CUDAExt === nothing && error("SHTnsKitGPUExt must load before its MPI composition extension")

const ParallelExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
ParallelExt === nothing && error("SHTnsKitParallelExt must load before its CUDA composition extension")

const VendorArray = CUDA.AnyCuArray
const VendorArrayType = CUDA.CuArray
const VENDOR_NAME = Val(:cuda)
include("ParallelGPUVendorFirewall.jl")

function _cuda_pinned(::Type{T}, n::Integer) where {T}
    return CUDA.pin(Vector{T}(undef, n))
end

const CUDA_PARALLEL_ADAPTER = ParallelExt.ParallelGPUAdapter(
    :cuda,
    value -> value isa CUDA.AnyCuArray,
    _ -> CUDA.CuArray,
    _ -> CUDA.device(),
    _ -> MPI.has_cuda(),
    _ -> CUDA.synchronize(),
    _cuda_pinned,
    (host, device) -> copyto!(host, device),
    (device, host) -> copyto!(device, host),
)

function __init__()
    ParallelExt._register_parallel_gpu_adapter!(CUDA_PARALLEL_ADAPTER)
end

@inline _first_m(plan) =
    first(PencilArrays.range_local(PencilArrays.pencil(plan.F_buf))[1]) - 1

function ParallelExt._dist_transpose_gpu_analysis!(
        ::Val{:cuda}, plan, output, input)
    LinearAlgebra.mul!(plan.F_buf, plan.fft_plan, input)
    RT = typeof(real(zero(eltype(output))))
    tables = CUDAExt._cuda_scalar_tables(plan.cfg, RT)
    kernel! = CUDAExt.GPUCommon.distributed_scalar_analysis_kernel!(
        CUDA.CUDABackend(),
    )
    kernel!(parent(output), parent(plan.F_buf), tables.Plm, tables.weights,
            RT(plan.cfg.cphi), _first_m(plan), plan.lmax, plan.mmax,
            plan.cfg.mres, plan.lmax;
            ndrange=size(parent(output)))
    CUDA.synchronize()
    return output
end

function ParallelExt._dist_transpose_gpu_synthesis!(
        ::Val{:cuda}, plan, output, input)
    fill!(parent(plan.F_buf), zero(eltype(plan.F_buf)))
    RT = typeof(real(zero(eltype(input))))
    tables = CUDAExt._cuda_scalar_tables(plan.cfg, RT)
    kernel! = CUDAExt.GPUCommon.distributed_scalar_synthesis_kernel!(
        CUDA.CUDABackend(),
    )
    kernel!(parent(plan.F_buf), parent(input), tables.Plm,
            RT(SHTnsKit.phi_inv_scale(plan.cfg)), _first_m(plan),
            plan.lmax, plan.mmax, plan.cfg.mres;
            ndrange=size(parent(plan.F_buf)))
    CUDA.synchronize()
    LinearAlgebra.ldiv!(output, plan.fft_plan, plan.F_buf)
    return output
end

function ParallelExt._dist_transpose_gpu_vector_analysis!(
        ::Val{:cuda}, plan, Sout, Tout, Vt, Vp)
    LinearAlgebra.mul!(plan.F_buf, plan.fft_plan, Vt)
    LinearAlgebra.mul!(plan.F_buf2, plan.fft_plan, Vp)
    RT = typeof(real(zero(eltype(Sout))))
    tables = CUDAExt._cuda_vector_tables(plan.cfg, RT)
    kernel! = CUDAExt.GPUCommon.distributed_vector_analysis_kernel!(
        CUDA.CUDABackend(),
    )
    kernel!(parent(Sout), parent(Tout), parent(plan.F_buf),
            parent(plan.F_buf2), tables.dtheta, tables.over_sin,
            tables.weights, tables.x, RT(plan.cfg.cphi), _first_m(plan),
            plan.lmax, plan.mmax, plan.cfg.mres, plan.cfg.robert_form;
            ndrange=size(parent(Sout)))
    CUDA.synchronize()
    return Sout, Tout
end

function ParallelExt._dist_transpose_gpu_vector_synthesis!(
        ::Val{:cuda}, plan, Vt, Vp, Sin, Tin)
    fill!(parent(plan.F_buf), zero(eltype(plan.F_buf)))
    fill!(parent(plan.F_buf2), zero(eltype(plan.F_buf2)))
    RT = typeof(real(zero(eltype(Sin))))
    tables = CUDAExt._cuda_vector_tables(plan.cfg, RT)
    kernel! = CUDAExt.GPUCommon.distributed_vector_synthesis_kernel!(
        CUDA.CUDABackend(),
    )
    kernel!(parent(plan.F_buf), parent(plan.F_buf2), parent(Sin), parent(Tin),
            tables.dtheta, tables.over_sin, tables.x,
            RT(SHTnsKit.phi_inv_scale(plan.cfg)), _first_m(plan),
            plan.lmax, plan.mmax, plan.cfg.mres, plan.cfg.robert_form;
            ndrange=size(parent(plan.F_buf)))
    CUDA.synchronize()
    LinearAlgebra.ldiv!(Vt, plan.fft_plan, plan.F_buf)
    LinearAlgebra.ldiv!(Vp, plan.fft_plan, plan.F_buf2)
    return Vt, Vp
end

end
