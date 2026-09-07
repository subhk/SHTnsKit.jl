# Exercise the production wrapper functions without vendor hardware. Only array
# allocation, device selection and FFT execution are modeled; the wrapper bodies
# and shared transform kernels are loaded directly from ext/.
module GPUWrapperReference

using Test, SHTnsKit, KernelAbstractions
include("../../ext/GPUCommon.jl")
include("test_mres.jl")

module HostVendor
    using KernelAbstractions
    const AnyROCArray = Array
    const AnyCuArray = Array
    const ROCArray = Array
    const CuArray = Array
    functional(args...) = true
    device() = 0
    device_id() = 0
    deviceid(device) = device
    synchronize() = KernelAbstractions.synchronize(KernelAbstractions.CPU())
    zeros(args...) = Base.zeros(args...)
end

module HostFFT
    using SHTnsKit
    const FFTW = SHTnsKit.FFTW
    # rocFFT implements * for in-place plans and mul! only for out-of-place
    # plans. Deliberately provide no mul! method for this wrapper.
    struct InplacePlan{P}
        plan::P
    end
    Base.:*(plan::InplacePlan, values::Array) = plan.plan * values
    plan_fft!(args...) = InplacePlan(FFTW.plan_fft!(args...))
    plan_ifft!(args...) = InplacePlan(FFTW.plan_ifft!(args...))
    plan_rfft(args...) = FFTW.plan_rfft(args...)
    plan_irfft(args...) = FFTW.plan_irfft(args...)
    fft!(args...) = FFTW.fft!(args...)
    ifft!(args...) = FFTW.ifft!(args...)
end

function function_name(expr)
    signature = expr.args[1]
    while signature isa Expr && signature.head === :where
        signature = signature.args[1]
    end
    return signature isa Expr && signature.head === :call ? signature.args[1] : nothing
end

function wrapper_module(vendor)
    prefix = vendor === :AMDGPU ? "amdgpu" : "cuda"
    filename = vendor === :AMDGPU ? "SHTnsKitAMDGPUExt.jl" : "SHTnsKitGPUExt.jl"
    source = Meta.parse(read(joinpath(@__DIR__, "../../ext", filename), String))
    sandbox = Module(gensym(:VendorWrapper))
    Core.eval(sandbox, :(const GPUCommon = $GPUCommon))
    Core.eval(sandbox, :(const $vendor = $HostVendor))
    Core.eval(sandbox, :(const FFTW = $(vendor === :AMDGPU ? HostFFT : SHTnsKit.FFTW)))
    Core.eval(sandbox, :(const CUFFT = $(SHTnsKit.FFTW)))
    Core.eval(sandbox, :(const gpu_ifft! = $(HostFFT.ifft!)))
    Core.eval(sandbox, :(const gpu_fft! = $(HostFFT.fft!)))
    Core.eval(sandbox, :(const ROCArray = Array))
    Core.eval(sandbox, :(const CuArray = Array))
    Core.eval(sandbox, :(const ROCBackend = $(KernelAbstractions.CPU)))
    Core.eval(sandbox, :(const CUDABackend = $(KernelAbstractions.CPU)))
    # Deterministically simulate stale allocator memory. A wrapper must clear
    # Fourier bins it does not write before passing them to the inverse FFT.
    Core.eval(sandbox, :(const similar = (args...) -> fill!(Base.similar(args...), NaN)))
    functions = Set(Symbol.("_" .* prefix .* [
        "_scalar_tables", "_vector_tables", "_workspace_builder",
        "_vector_workspace_builder", "_batch_scratch",
        "_scalar_analysis_direct!", "_scalar_synthesis_direct!",
        "_batch_analysis_direct!", "_batch_synthesis_direct!",
        "_vector_analysis_direct!", "_vector_synthesis_direct!",
        "_vector_batch_synthesis",
    ]))
    union!(functions, [Symbol("_with_", prefix, "_workspace"),
                       Symbol("_with_", prefix, "_vector_workspace"),
                       Symbol("_require_", prefix)])
    typeprefix = vendor === :AMDGPU ? "AMDGPU" : "CUDA"
    structs = Set(Symbol.(typeprefix .* ["ScalarTables", "VectorTables"]))
    constants = Set(Symbol.("_" .* typeprefix .* [
        "_SCALAR_CACHE", "_VECTOR_CACHE", "_WORKSPACE_CACHE",
    ]))
    for expr in source.args[3].args
        expr isa Expr || continue
        if expr.head === :using
            # Preserve the production imports (including LinearAlgebra) so a
            # missing mul! binding is caught, while replacing vendor/FFT APIs.
            for imported in expr.args
                imported.head === :. && first(imported.args) in (:AMDGPU, :CUDA, :FFTW) && continue
                Core.eval(sandbox, Expr(:using, imported))
            end
        elseif expr.head === :struct
            declaration = expr.args[2]
            name = declaration isa Expr ? declaration.args[1] : declaration
            name in structs && Core.eval(sandbox, expr)
        elseif expr.head === :const
            expr.args[1].args[1] in constants && Core.eval(sandbox, expr)
        elseif expr.head === :function && function_name(expr) in functions
            Core.eval(sandbox, expr)
        end
    end
    return sandbox
end

@testset "GPU wrapper host reference" begin
    rocm = wrapper_module(:AMDGPU)
    cuda = wrapper_module(:CUDA)
    @testset "Order stride through production wrappers" begin
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda"))
            scalar_analysis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_analysis_direct!"))
            scalar_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_synthesis_direct!"))
            vector_analysis! = getproperty(wrapper, Symbol("_", prefix, "_vector_analysis_direct!"))
            vector_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_vector_synthesis_direct!"))
            run_gpu_mres_tests(
                scalar_analysis=(cfg, field) -> begin
                    output = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
                    scalar_analysis!(cfg, cfg, output, field)
                end,
                scalar_synthesis=(cfg, coefficients; real_output=true) -> begin
                    output = zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon)
                    scalar_synthesis!(cfg, cfg, output, coefficients; real_output)
                end,
                vector_analysis=(cfg, vt, vp) -> begin
                    sout = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
                    tout = similar(sout)
                    vector_analysis!(cfg, cfg, sout, tout, vt, vp)
                end,
                vector_synthesis=(cfg, s, t; real_output=true) -> begin
                    vt = zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon)
                    vp = similar(vt)
                    vector_synthesis!(cfg, cfg, vt, vp, s, t; real_output)
                end,
            )
        end
    end
    @testset "Planned FFT execution" begin
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda")),
            T in (Float32, Float64), use_rfft in (false, true)
            scalar_analysis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_analysis_direct!"))
            scalar_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_synthesis_direct!"))
            batch_analysis! = getproperty(wrapper, Symbol("_", prefix, "_batch_analysis_direct!"))
            batch_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_batch_synthesis_direct!"))
            vector_analysis! = getproperty(wrapper, Symbol("_", prefix, "_vector_analysis_direct!"))
            vector_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_vector_synthesis_direct!"))
            cfg = create_gauss_config(3, 8; nlon=10)
            coeff = zeros(Complex{T}, 4, 4)
            coeff[2, 1] = T(0.25)
            coeff[3, 3] = Complex{T}(0.1, 0.2)
            field = synthesis(cfg, coeff)
            output = similar(coeff)
            rebuilt = similar(field)
            tol = T === Float32 ? 3e-5 : 2e-12
            @test scalar_analysis!(cfg, cfg, output, field; use_rfft) === output
            @test output ≈ coeff atol=tol rtol=tol
            @test scalar_synthesis!(cfg, cfg, rebuilt, coeff; use_rfft) === rebuilt
            @test rebuilt ≈ field atol=tol rtol=tol
            fields = cat(field, 2field; dims=3)
            coefficients = cat(coeff, 2coeff; dims=3)
            batchout = similar(coefficients)
            spatialout = similar(fields)
            bins = use_rfft ? cfg.nlon ÷ 2 + 1 : cfg.nlon
            scratch = zeros(Complex{T}, cfg.nlat, bins, 2)
            for fft_batch in (nothing, scratch)
                @test batch_analysis!(cfg, batchout, fields; use_rfft, fft_batch) === batchout
                @test batchout ≈ coefficients atol=tol rtol=tol
                @test batch_synthesis!(cfg, spatialout, coefficients; use_rfft, fft_batch) === spatialout
                @test spatialout ≈ fields atol=tol rtol=tol
            end
            vt, vp = synthesis_sphtor(cfg, coeff, 2coeff)
            sout, tout = similar(coeff), similar(coeff)
            @test vector_analysis!(cfg, cfg, sout, tout, vt, vp; use_rfft) === (sout, tout)
            @test sout ≈ coeff atol=tol rtol=tol
            @test tout ≈ 2coeff atol=tol rtol=tol
            vtout, vpout = similar(vt), similar(vp)
            @test vector_synthesis!(cfg, cfg, vtout, vpout, coeff, 2coeff; use_rfft) === (vtout, vpout)
            @test vtout ≈ vt atol=tol rtol=tol
            @test vpout ≈ vp atol=tol rtol=tol
        end
    end
    @testset "Vector batch synthesis clears unused Fourier bins" begin
        for (vendor, call) in ((:AMDGPU, rocm._amdgpu_vector_batch_synthesis),
                               (:CUDA, cuda._cuda_vector_batch_synthesis)),
            T in (Float32, Float64), mres in (1, 2), real_output in (false, true)
            cfg = create_gauss_config(3, 8; nlon=10, mres)
            coeff = zeros(Complex{T}, 4, 4, 2)
            for nonzero in (false, true)
                if nonzero
                    coeff[2, 1, 1] = T(0.25)
                    coeff[3, 3, 2] = Complex{T}(0.1, 0.2)
                end
                expected = synthesis_sphtor_batch(cfg, coeff, 2coeff; real_output)
                result = call(cfg, coeff, 2coeff; real_output)
                tol = T === Float32 ? 3e-5 : 2e-12
                @test result[1] ≈ expected[1] atol=tol rtol=tol
                @test result[2] ≈ expected[2] atol=tol rtol=tol
            end
        end
    end
end

end # module
