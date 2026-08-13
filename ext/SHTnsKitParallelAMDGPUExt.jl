module SHTnsKitParallelAMDGPUExt

import SHTnsKit
import MPI
import PencilArrays
import PencilFFTs
import AMDGPU
import GPUArrays
import GPUArraysCore
import KernelAbstractions
import LinearAlgebra

const AMDGPUExt = Base.get_extension(SHTnsKit, :SHTnsKitAMDGPUExt)
AMDGPUExt === nothing && error("SHTnsKitAMDGPUExt must load before its MPI composition extension")

const ParallelExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
ParallelExt === nothing && error("SHTnsKitParallelExt must load before its AMDGPU composition extension")

const VendorArray = AMDGPU.AnyROCArray
const VendorArrayType = AMDGPU.ROCArray
const VENDOR_NAME = Val(:amdgpu)
include("ParallelGPUVendorFirewall.jl")

function _amdgpu_pinned(::Type{T}, n::Integer) where {T}
    host = Vector{T}(undef, n)
    AMDGPU.Mem.pin(pointer(host), sizeof(host))
    finalizer(host) do value
        try
            AMDGPU.Mem.unpin(pointer(value))
        catch
        end
    end
    return host
end

const AMDGPU_PARALLEL_ADAPTER = ParallelExt.ParallelGPUAdapter(
    :amdgpu,
    value -> ParallelExt._parallel_root_buffer(value) isa AMDGPU.AnyROCArray,
    _ -> AMDGPU.ROCArray,
    value -> AMDGPU.device_id(AMDGPU.device(
        ParallelExt._parallel_root_buffer(value),
    )),
    _ -> MPI.has_rocm(),
    _ -> AMDGPU.synchronize(),
    _amdgpu_pinned,
    (host, device) -> copyto!(host, device),
    (device, host) -> copyto!(device, host),
)

function __init__()
    ParallelExt._register_parallel_gpu_adapter!(AMDGPU_PARALLEL_ADAPTER)
end

@inline _first_m(plan) =
    first(PencilArrays.range_local(PencilArrays.pencil(plan.F_buf))[1]) - 1

function ParallelExt._dist_transpose_gpu_analysis!(
        ::Val{:amdgpu}, plan, output, input)
    LinearAlgebra.mul!(plan.F_buf, plan.fft_plan, input)
    RT = typeof(real(zero(eltype(output))))
    tables = AMDGPUExt._amdgpu_scalar_tables(plan.cfg, RT)
    kernel! = AMDGPUExt.GPUCommon.distributed_scalar_analysis_kernel!(
        AMDGPU.ROCBackend(),
    )
    kernel!(parent(output), parent(plan.F_buf), tables.Plm, tables.weights,
            RT(plan.cfg.cphi), _first_m(plan), plan.lmax, plan.mmax,
            plan.cfg.mres, plan.lmax;
            ndrange=size(parent(output)))
    AMDGPU.synchronize()
    return output
end

function ParallelExt._dist_transpose_gpu_synthesis!(
        ::Val{:amdgpu}, plan, output, input)
    fill!(parent(plan.F_buf), zero(eltype(plan.F_buf)))
    RT = typeof(real(zero(eltype(input))))
    tables = AMDGPUExt._amdgpu_scalar_tables(plan.cfg, RT)
    kernel! = AMDGPUExt.GPUCommon.distributed_scalar_synthesis_kernel!(
        AMDGPU.ROCBackend(),
    )
    kernel!(parent(plan.F_buf), parent(input), tables.Plm,
            RT(SHTnsKit.phi_inv_scale(plan.cfg)), _first_m(plan),
            plan.lmax, plan.mmax, plan.cfg.mres;
            ndrange=size(parent(plan.F_buf)))
    AMDGPU.synchronize()
    LinearAlgebra.ldiv!(output, plan.fft_plan, plan.F_buf)
    return output
end

function ParallelExt._dist_transpose_gpu_vector_analysis!(
        ::Val{:amdgpu}, plan, Sout, Tout, Vt, Vp)
    LinearAlgebra.mul!(plan.F_buf, plan.fft_plan, Vt)
    LinearAlgebra.mul!(plan.F_buf2, plan.fft_plan, Vp)
    RT = typeof(real(zero(eltype(Sout))))
    tables = AMDGPUExt._amdgpu_vector_tables(plan.cfg, RT)
    kernel! = AMDGPUExt.GPUCommon.distributed_vector_analysis_kernel!(
        AMDGPU.ROCBackend(),
    )
    kernel!(parent(Sout), parent(Tout), parent(plan.F_buf),
            parent(plan.F_buf2), tables.dtheta, tables.over_sin,
            tables.weights, tables.x, RT(plan.cfg.cphi), _first_m(plan),
            plan.lmax, plan.mmax, plan.cfg.mres, plan.cfg.robert_form;
            ndrange=size(parent(Sout)))
    AMDGPU.synchronize()
    return Sout, Tout
end

function ParallelExt._dist_transpose_gpu_vector_synthesis!(
        ::Val{:amdgpu}, plan, Vt, Vp, Sin, Tin)
    fill!(parent(plan.F_buf), zero(eltype(plan.F_buf)))
    fill!(parent(plan.F_buf2), zero(eltype(plan.F_buf2)))
    RT = typeof(real(zero(eltype(Sin))))
    tables = AMDGPUExt._amdgpu_vector_tables(plan.cfg, RT)
    kernel! = AMDGPUExt.GPUCommon.distributed_vector_synthesis_kernel!(
        AMDGPU.ROCBackend(),
    )
    kernel!(parent(plan.F_buf), parent(plan.F_buf2), parent(Sin), parent(Tin),
            tables.dtheta, tables.over_sin, tables.x,
            RT(SHTnsKit.phi_inv_scale(plan.cfg)), _first_m(plan),
            plan.lmax, plan.mmax, plan.cfg.mres, plan.cfg.robert_form;
            ndrange=size(parent(plan.F_buf)))
    AMDGPU.synchronize()
    LinearAlgebra.ldiv!(Vt, plan.fft_plan, plan.F_buf)
    LinearAlgebra.ldiv!(Vp, plan.fft_plan, plan.F_buf2)
    return Vt, Vp
end

end
