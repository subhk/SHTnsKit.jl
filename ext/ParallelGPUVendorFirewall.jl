##########
# Vendor-owned PencilArray dispatch firewall
##########

# This file is included inside each compound MPI+GPU extension after it defines
# `VendorArray`, `VendorArrayType`, and `VENDOR_NAME`.  Methods added here are
# therefore owned by the compound extension and are more specific than the
# CPU/MPI `PencilArray` methods in `SHTnsKitParallelExt`.

const VendorPencilArray = PencilArrays.PencilArray{
    T,N,A,Np,E,P,
} where {T,N,A<:VendorArray,Np,E,P}

@inline _vendor_parent(value::VendorPencilArray) = PencilArrays.parent(value)
@inline _vendor_parent(value) = value

@inline function _vendor_comm(values...)
    for value in values
        value isa VendorPencilArray && return PencilArrays.communicator(value)
    end
    throw(ArgumentError("MPI GPU call requires a vendor-backed PencilArray"))
end

@inline _vendor_adapter(value::VendorPencilArray) =
    ParallelExt._parallel_gpu_adapter(PencilArrays.parent(value))

function _stage_vendor_call(operation::Symbol, f, values...;
                            mutated::Tuple=())
    prototype = first(value for value in values if value isa VendorPencilArray)
    return ParallelExt._staged_gpu_call(
        _vendor_adapter(prototype), operation, _vendor_comm(values...), f,
        values...; mutated,
    )
end

# The ordinary full-grid APIs already have device-native transpose-plan entry
# points in the parallel extension and the vendor-specific kernels below this
# include.  The remaining cfg-form APIs are deliberately staged at this single
# compound-extension boundary, before any generic CPU PencilArray method can
# index device storage.

function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig,
                           field::VendorPencilArray; kwargs...)
    return _stage_vendor_call(:analysis, field) do host
        SHTnsKit.analysis(cfg, host; kwargs...)
    end
end

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig,
                            coefficients::VendorPencilArray;
                            prototype_θφ::VendorPencilArray, kwargs...)
    return _stage_vendor_call(
        :synthesis,
        (host_coefficients, host_prototype) -> SHTnsKit.synthesis(
            cfg, host_coefficients; prototype_θφ=host_prototype, kwargs...,
        ),
        coefficients, prototype_θφ,
    )
end

function SHTnsKit.synthesis_cplx(cfg::SHTnsKit.SHTConfig,
                                 coefficients::VendorPencilArray;
                                 prototype_θφ::VendorPencilArray)
    return _stage_vendor_call(
        :synthesis_cplx,
        (host_coefficients, host_prototype) -> SHTnsKit.synthesis_cplx(
            cfg, host_coefficients; prototype_θφ=host_prototype,
        ), coefficients, prototype_θφ,
    )
end

function SHTnsKit.analysis_sphtor(cfg::SHTnsKit.SHTConfig,
                                  Vt::VendorPencilArray,
                                  Vp::VendorPencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_sphtor,
        (host_vt, host_vp) -> SHTnsKit.analysis_sphtor(
            cfg, host_vt, host_vp; kwargs...,
        ),
        Vt, Vp,
    )
end

function SHTnsKit.synthesis_sphtor(cfg::SHTnsKit.SHTConfig,
                                   S::VendorPencilArray,
                                   T::VendorPencilArray;
                                   prototype_θφ::VendorPencilArray, kwargs...)
    return _stage_vendor_call(
        :synthesis_sphtor,
        (host_s, host_t, host_prototype) -> SHTnsKit.synthesis_sphtor(
            cfg, host_s, host_t; prototype_θφ=host_prototype, kwargs...,
        ),
        S, T, prototype_θφ,
    )
end

function SHTnsKit.analysis_qst(cfg::SHTnsKit.SHTConfig,
                               Vr::VendorPencilArray,
                               Vt::VendorPencilArray,
                               Vp::VendorPencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_qst,
        (host_vr, host_vt, host_vp) -> SHTnsKit.analysis_qst(
            cfg, host_vr, host_vt, host_vp; kwargs...,
        ),
        Vr, Vt, Vp,
    )
end

function SHTnsKit.synthesis_qst(cfg::SHTnsKit.SHTConfig,
                                Q::VendorPencilArray,
                                S::VendorPencilArray,
                                T::VendorPencilArray;
                                prototype_θφ::VendorPencilArray, kwargs...)
    return _stage_vendor_call(
        :synthesis_qst,
        (host_q, host_s, host_t, host_prototype) -> SHTnsKit.synthesis_qst(
            cfg, host_q, host_s, host_t;
            prototype_θφ=host_prototype, kwargs...,
        ),
        Q, S, T, prototype_θφ,
    )
end

# Same-shape local spectral operations.  Out-of-place results are restored to
# the device; bang variants copy the staged destination back and return the
# original object.
for name in (
        :divergence_from_spheroidal, :spheroidal_from_divergence,
        :vorticity_from_toroidal, :toroidal_from_vorticity,
        :energy_scalar, :energy_scalar_l_spectrum,
        :energy_scalar_m_spectrum, :grid_energy_scalar, :grid_enstrophy,
    )
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

for name in (
        :divergence_from_spheroidal!, :spheroidal_from_divergence!,
        :vorticity_from_toroidal!, :toroidal_from_vorticity!,
    )
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  output::VendorPencilArray,
                                  input::VendorPencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_output, host_input) -> SHTnsKit.$name(
                cfg, host_output, host_input,
            ),
            output, input; mutated=(1,),
        )
    end
end

function SHTnsKit.dist_apply_laplacian!(cfg::SHTnsKit.SHTConfig,
                                        coefficients::VendorPencilArray)
    return _stage_vendor_call(
        :dist_apply_laplacian!,
        host -> SHTnsKit.dist_apply_laplacian!(cfg, host), coefficients;
        mutated=(1,),
    )
end

# Native transpose-plan entry points.  These stay entirely device-resident:
# PencilFFTs owns the distributed FFT/transpose and the compound extension owns
# the vendor Legendre kernels selected by these helpers.
function SHTnsKit.dist_analysis!(plan::ParallelExt.DistTransposePlan,
                                 output::VendorPencilArray,
                                 input::VendorPencilArray)
    ParallelExt._validate_transpose_call!(
        plan, :dist_analysis_transpose; spatial=(input,), spectral=(output,),
    )
    return ParallelExt._dist_transpose_gpu_analysis!(
        _vendor_adapter(input), plan, output, input,
    )
end

function SHTnsKit.dist_synthesis!(plan::ParallelExt.DistTransposePlan,
                                  output::VendorPencilArray,
                                  input::VendorPencilArray)
    ParallelExt._validate_transpose_call!(
        plan, :dist_synthesis_transpose; spatial=(output,), spectral=(input,),
    )
    return ParallelExt._dist_transpose_gpu_synthesis!(
        _vendor_adapter(output), plan, output, input,
    )
end

function SHTnsKit.dist_analysis_sphtor!(plan::ParallelExt.DistTransposePlan,
                                        S::VendorPencilArray,
                                        T::VendorPencilArray,
                                        Vt::VendorPencilArray,
                                        Vp::VendorPencilArray)
    ParallelExt._validate_transpose_call!(
        plan, :dist_analysis_sphtor_transpose;
        spatial=(Vt, Vp), spectral=(S, T),
    )
    plan.with_vector || throw(ArgumentError(
        "DistTransposePlan requires with_vector=true for sphtor transforms",
    ))
    return ParallelExt._dist_transpose_gpu_vector_analysis!(
        _vendor_adapter(Vt), plan, S, T, Vt, Vp,
    )
end

function SHTnsKit.dist_synthesis_sphtor!(plan::ParallelExt.DistTransposePlan,
                                         Vt::VendorPencilArray,
                                         Vp::VendorPencilArray,
                                         S::VendorPencilArray,
                                         T::VendorPencilArray)
    ParallelExt._validate_transpose_call!(
        plan, :dist_synthesis_sphtor_transpose;
        spatial=(Vt, Vp), spectral=(S, T),
    )
    plan.with_vector || throw(ArgumentError(
        "DistTransposePlan requires with_vector=true for sphtor transforms",
    ))
    return ParallelExt._dist_transpose_gpu_vector_synthesis!(
        _vendor_adapter(Vt), plan, Vt, Vp, S, T,
    )
end

function SHTnsKit.dist_analysis_qst!(plan::ParallelExt.DistTransposePlan,
                                     Q::VendorPencilArray,
                                     S::VendorPencilArray,
                                     T::VendorPencilArray,
                                     Vr::VendorPencilArray,
                                     Vt::VendorPencilArray,
                                     Vp::VendorPencilArray)
    SHTnsKit.dist_analysis!(plan, Q, Vr)
    SHTnsKit.dist_analysis_sphtor!(plan, S, T, Vt, Vp)
    return Q, S, T
end

function SHTnsKit.dist_synthesis_qst!(plan::ParallelExt.DistTransposePlan,
                                      Vr::VendorPencilArray,
                                      Vt::VendorPencilArray,
                                      Vp::VendorPencilArray,
                                      Q::VendorPencilArray,
                                      S::VendorPencilArray,
                                      T::VendorPencilArray)
    SHTnsKit.dist_synthesis!(plan, Vr, Q)
    SHTnsKit.dist_synthesis_sphtor!(plan, Vt, Vp, S, T)
    return Vr, Vt, Vp
end

# Complex aliases and degree/order-restricted transforms.
for name in (:analysis_sphtor_cplx,)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  Vt::VendorPencilArray,
                                  Vp::VendorPencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_vt, host_vp) -> SHTnsKit.$name(
                cfg, host_vt, host_vp; kwargs...,
            ), Vt, Vp,
        )
    end
end

for name in (:synthesis_sphtor_cplx,)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  S::VendorPencilArray,
                                  T::VendorPencilArray;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_s, host_t, host_prototype) -> SHTnsKit.$name(
                cfg, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), S, T, prototype_θφ,
        )
    end
end

for name in (:synthesis_sph, :synthesis_sph_cplx,
             :synthesis_tor, :synthesis_tor_cplx, :synthesis_grad)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients;
                prototype_θφ=host_prototype, kwargs...,
            ), coefficients, prototype_θφ,
        )
    end
end

function SHTnsKit.analysis_sphtor_l(cfg::SHTnsKit.SHTConfig,
                                    Vt::VendorPencilArray,
                                    Vp::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(
        :analysis_sphtor_l,
        (host_vt, host_vp) -> SHTnsKit.analysis_sphtor_l(
            cfg, host_vt, host_vp, ltr,
        ), Vt, Vp,
    )
end

for name in (:synthesis_sphtor_l, :synthesis_sphtor_l_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  S::VendorPencilArray,
                                  T::VendorPencilArray, ltr::Integer;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_s, host_t, host_prototype) -> SHTnsKit.$name(
                cfg, host_s, host_t, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), S, T, prototype_θφ,
        )
    end
end

for name in (:synthesis_sph_l, :synthesis_tor_l, :synthesis_grad_l)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray,
                                  ltr::Integer;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), coefficients, prototype_θφ,
        )
    end
end

function SHTnsKit.analysis_sphtor_ml(cfg::SHTnsKit.SHTConfig,
                                     stored_im::Integer,
                                     Vt::VendorPencilArray,
                                     Vp::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(
        :analysis_sphtor_ml,
        (host_vt, host_vp) -> SHTnsKit.analysis_sphtor_ml(
            cfg, stored_im, host_vt, host_vp, ltr,
        ), Vt, Vp,
    )
end

function SHTnsKit.synthesis_sphtor_ml(cfg::SHTnsKit.SHTConfig,
                                      stored_im::Integer,
                                      S::VendorPencilArray,
                                      T::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(
        :synthesis_sphtor_ml,
        (host_s, host_t) -> SHTnsKit.synthesis_sphtor_ml(
            cfg, stored_im, host_s, host_t, ltr,
        ), S, T,
    )
end

for name in (:synthesis_sph_ml, :synthesis_tor_ml, :synthesis_grad_ml)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  stored_im::Integer,
                                  coefficients::VendorPencilArray,
                                  ltr::Integer)
        return _stage_vendor_call($(QuoteNode(name)), coefficients) do host
            SHTnsKit.$name(cfg, stored_im, host, ltr)
        end
    end
end

function SHTnsKit.analysis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                    Vr::VendorPencilArray,
                                    Vt::VendorPencilArray,
                                    Vp::VendorPencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_qst_cplx,
        (host_vr, host_vt, host_vp) -> SHTnsKit.analysis_qst_cplx(
            cfg, host_vr, host_vt, host_vp; kwargs...,
        ), Vr, Vt, Vp,
    )
end

function SHTnsKit.analysis_qst_l(cfg::SHTnsKit.SHTConfig,
                                 Vr::VendorPencilArray,
                                 Vt::VendorPencilArray,
                                 Vp::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(
        :analysis_qst_l,
        (host_vr, host_vt, host_vp) -> SHTnsKit.analysis_qst_l(
            cfg, host_vr, host_vt, host_vp, ltr,
        ), Vr, Vt, Vp,
    )
end

function SHTnsKit.synthesis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                     Q::VendorPencilArray,
                                     S::VendorPencilArray,
                                     T::VendorPencilArray;
                                     prototype_θφ::VendorPencilArray)
    return _stage_vendor_call(
        :synthesis_qst_cplx,
        (host_q, host_s, host_t, host_prototype) ->
            SHTnsKit.synthesis_qst_cplx(
                cfg, host_q, host_s, host_t;
                prototype_θφ=host_prototype,
            ), Q, S, T, prototype_θφ,
    )
end


for name in (:synthesis_qst_l, :synthesis_qst_l_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  Q::VendorPencilArray,
                                  S::VendorPencilArray,
                                  T::VendorPencilArray, ltr::Integer;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_q, host_s, host_t, host_prototype) -> SHTnsKit.$name(
                cfg, host_q, host_s, host_t, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), Q, S, T, prototype_θφ,
        )
    end
end

for name in (:analysis_qst_ml, :synthesis_qst_ml)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  stored_im::Integer,
                                  Q::VendorPencilArray,
                                  S::VendorPencilArray,
                                  T::VendorPencilArray, ltr::Integer)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_q, host_s, host_t) -> SHTnsKit.$name(
                cfg, stored_im, host_q, host_s, host_t, ltr,
            ), Q, S, T,
        )
    end
end

# Packed, axisymmetric, batch, and local-evaluation families.
for name in (:analysis_packed, :analysis_packed_cplx, :analysis_axisym)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

for name in (:analysis_packed_l, :analysis_packed_cplx_l, :analysis_axisym_l)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, ltr::Integer;
                                  kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host, ltr; kwargs...)
        end
    end
end

for name in (:synthesis_packed, :synthesis_packed_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_prototype) -> SHTnsKit.$name(
                cfg, host_input;
                prototype_θφ=host_prototype, kwargs...,
            ), input, prototype_θφ,
        )
    end
end

for name in (:synthesis_packed_l, :synthesis_packed_cplx_l)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, ltr::Integer;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_prototype) -> SHTnsKit.$name(
                cfg, host_input, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), input, prototype_θφ,
        )
    end
end

function SHTnsKit.synthesis_axisym(cfg::SHTnsKit.SHTConfig,
                                   input::VendorPencilArray)
    return _stage_vendor_call(:synthesis_axisym, input) do host
        SHTnsKit.synthesis_axisym(cfg, host)
    end
end

function SHTnsKit.synthesis_axisym_l(cfg::SHTnsKit.SHTConfig,
                                     input::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(:synthesis_axisym_l, input) do host
        SHTnsKit.synthesis_axisym_l(cfg, host, ltr)
    end
end

function SHTnsKit.analysis_packed_ml(cfg::SHTnsKit.SHTConfig,
                                     stored_im::Int,
                                     field::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(:analysis_packed_ml, field) do host
        SHTnsKit.analysis_packed_ml(cfg, stored_im, host, ltr)
    end
end

function SHTnsKit.synthesis_packed_ml(cfg::SHTnsKit.SHTConfig,
                                      stored_im::Int,
                                      coefficients::VendorPencilArray,
                                      ltr::Integer)
    return _stage_vendor_call(:synthesis_packed_ml, coefficients) do host
        SHTnsKit.synthesis_packed_ml(cfg, stored_im, host, ltr)
    end
end

for name in (:analysis_batch,)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

function SHTnsKit.analysis_batch!(cfg::SHTnsKit.SHTConfig,
                                   output::VendorPencilArray,
                                   input::VendorPencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_batch!,
        (host_output, host_input) -> SHTnsKit.analysis_batch!(
            cfg, host_output, host_input; kwargs...,
        ), output, input; mutated=(1,),
    )
end

for name in (:synthesis_batch, :synthesis_batch_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray;
                                  prototype_θφ::VendorPencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_prototype) -> SHTnsKit.$name(
                cfg, host_input; prototype_θφ=host_prototype, kwargs...,
            ), input, prototype_θφ,
        )
    end
end

function SHTnsKit.synthesis_batch!(cfg::SHTnsKit.SHTConfig,
                                    output::VendorPencilArray,
                                    input::VendorPencilArray;
                                    prototype_θφ::VendorPencilArray=output,
                                    kwargs...)
    return _stage_vendor_call(
        :synthesis_batch!,
        (host_output, host_input, host_prototype) ->
            SHTnsKit.synthesis_batch!(
                cfg, host_output, host_input;
                prototype_θφ=host_prototype, kwargs...,
            ),
        output, input, prototype_θφ; mutated=(1,),
    )
end

for name in (:analysis_sphtor_batch, :synthesis_sphtor_batch,
             :synthesis_sphtor_batch_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::VendorPencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second) -> SHTnsKit.$name(
                cfg, host_first, host_second; kwargs...,
            ), first, second,
        )
    end
end

for name in (:analysis_qst_batch, :synthesis_qst_batch,
             :synthesis_qst_batch_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::VendorPencilArray,
                                  third::VendorPencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second, host_third) -> SHTnsKit.$name(
                cfg, host_first, host_second, host_third; kwargs...,
            ), first, second, third,
        )
    end
end

for name in (:synthesis_point, :synthesis_point_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray,
                                  cost::Real, phi::Real)
        return _stage_vendor_call($(QuoteNode(name)), coefficients) do host
            SHTnsKit.$name(cfg, host, cost, phi)
        end
    end
end

for name in (:SH_to_lat, :SH_to_lat_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray,
                                  cost::Real; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), coefficients) do host
            SHTnsKit.$name(cfg, host, cost; kwargs...)
        end
    end
end

for name in (:SHqst_to_point, :SH_to_grad_point)
    # SH_to_grad_point is covered by the two-input prefix of this family.
    if name === :SHqst_to_point
        @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                      Q::VendorPencilArray,
                                      S::VendorPencilArray,
                                      T::VendorPencilArray,
                                      cost::Real, phi::Real)
            return _stage_vendor_call(
                $(QuoteNode(name)),
                (host_q, host_s, host_t) -> SHTnsKit.$name(
                    cfg, host_q, host_s, host_t, cost, phi,
                ), Q, S, T,
            )
        end
    end
end

function SHTnsKit.SH_to_grad_point(cfg::SHTnsKit.SHTConfig,
                                   Dr::VendorPencilArray,
                                   S::VendorPencilArray,
                                   cost::Real, phi::Real)
    return _stage_vendor_call(
        :SH_to_grad_point,
        (host_dr, host_s) -> SHTnsKit.SH_to_grad_point(
            cfg, host_dr, host_s, cost, phi,
        ), Dr, S,
    )
end

function SHTnsKit.SHqst_to_lat(cfg::SHTnsKit.SHTConfig,
                               Q::VendorPencilArray,
                               S::VendorPencilArray,
                               T::VendorPencilArray, cost::Real; kwargs...)
    return _stage_vendor_call(
        :SHqst_to_lat,
        (host_q, host_s, host_t) -> SHTnsKit.SHqst_to_lat(
            cfg, host_q, host_s, host_t, cost; kwargs...,
        ), Q, S, T,
    )
end

# Preserved `dist_*` aliases use the same staged boundary for cfg-form calls.
function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig,
                                field::VendorPencilArray; kwargs...)
    return _stage_vendor_call(:dist_analysis, field) do host
        SHTnsKit.dist_analysis(cfg, host; kwargs...)
    end
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig,
                                 coefficients::VendorPencilArray;
                                 prototype_θφ::VendorPencilArray, kwargs...)
    return _stage_vendor_call(
        :dist_synthesis,
        (host_coefficients, host_prototype) -> SHTnsKit.dist_synthesis(
            cfg, host_coefficients;
            prototype_θφ=host_prototype, kwargs...,
        ), coefficients, prototype_θφ,
    )
end

function SHTnsKit.dist_analysis_sphtor(cfg::SHTnsKit.SHTConfig,
                                       Vt::VendorPencilArray,
                                       Vp::VendorPencilArray; kwargs...)
    return _stage_vendor_call(
        :dist_analysis_sphtor,
        (host_vt, host_vp) -> SHTnsKit.dist_analysis_sphtor(
            cfg, host_vt, host_vp; kwargs...,
        ), Vt, Vp,
    )
end

function SHTnsKit.dist_synthesis_sphtor(cfg::SHTnsKit.SHTConfig,
                                        S::VendorPencilArray,
                                        T::VendorPencilArray;
                                        prototype_θφ::VendorPencilArray,
                                        kwargs...)
    return _stage_vendor_call(
        :dist_synthesis_sphtor,
        (host_s, host_t, host_prototype) ->
            SHTnsKit.dist_synthesis_sphtor(
                cfg, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), S, T, prototype_θφ,
    )
end

function SHTnsKit.dist_analysis_qst(cfg::SHTnsKit.SHTConfig,
                                    Vr::VendorPencilArray,
                                    Vt::VendorPencilArray,
                                    Vp::VendorPencilArray; kwargs...)
    return _stage_vendor_call(
        :dist_analysis_qst,
        (host_vr, host_vt, host_vp) -> SHTnsKit.dist_analysis_qst(
            cfg, host_vr, host_vt, host_vp; kwargs...,
        ), Vr, Vt, Vp,
    )
end

function SHTnsKit.dist_synthesis_qst(cfg::SHTnsKit.SHTConfig,
                                     Q::VendorPencilArray,
                                     S::VendorPencilArray,
                                     T::VendorPencilArray;
                                     prototype_θφ::VendorPencilArray,
                                     kwargs...)
    return _stage_vendor_call(
        :dist_synthesis_qst,
        (host_q, host_s, host_t, host_prototype) ->
            SHTnsKit.dist_synthesis_qst(
                cfg, host_q, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), Q, S, T, prototype_θφ,
    )
end

function SHTnsKit.dist_scalar_roundtrip!(cfg::SHTnsKit.SHTConfig,
                                         field::VendorPencilArray)
    return _stage_vendor_call(:dist_scalar_roundtrip!, field) do host
        SHTnsKit.dist_scalar_roundtrip!(cfg, host)
    end
end

function SHTnsKit.dist_vector_roundtrip!(cfg::SHTnsKit.SHTConfig,
                                         Vt::VendorPencilArray,
                                         Vp::VendorPencilArray)
    return _stage_vendor_call(
        :dist_vector_roundtrip!,
        (host_vt, host_vp) -> SHTnsKit.dist_vector_roundtrip!(
            cfg, host_vt, host_vp,
        ), Vt, Vp,
    )
end

# Local operators, spatial composites, rotations, and diagnostics.
function SHTnsKit.SH_mul_mx(::SHTnsKit.CPU, cfg::SHTnsKit.SHTConfig,
                            mx::AbstractVector{<:Real},
                            input::VendorPencilArray,
                            output::VendorPencilArray)
    return _stage_vendor_call(
        :SH_mul_mx,
        (host_input, host_output) -> SHTnsKit.SH_mul_mx(
            SHTnsKit.CPU(), cfg, mx, host_input, host_output,
        ), input, output; mutated=(2,),
    )
end

SHTnsKit.SH_mul_mx(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real},
                   input::VendorPencilArray, output::VendorPencilArray) =
    SHTnsKit.SH_mul_mx(SHTnsKit.CPU(), cfg, mx, input, output)

SHTnsKit.dist_SH_mul_mx!(cfg::SHTnsKit.SHTConfig,
                         mx::AbstractVector{<:Real},
                         input::VendorPencilArray,
                         output::VendorPencilArray) =
    SHTnsKit.SH_mul_mx(SHTnsKit.CPU(), cfg, mx, input, output)

for name in (:dist_spatial_divergence, :dist_spatial_vorticity)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::VendorPencilArray; kwargs...)
        prototype = get(kwargs, :prototype_θφ, first)
        prototype isa VendorPencilArray || throw(ArgumentError(
            "$(string($name)) prototype must use the input GPU vendor",
        ))
        forwarded = Base.structdiff((; kwargs...), (; prototype_θφ=prototype))
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second, host_prototype) -> SHTnsKit.$name(
                cfg, host_first, host_second;
                prototype_θφ=host_prototype, forwarded...,
            ), first, second, prototype,
        )
    end
end

function SHTnsKit.dist_scalar_laplacian(cfg::SHTnsKit.SHTConfig,
                                        input::VendorPencilArray; kwargs...)
    prototype = get(kwargs, :prototype_θφ, input)
    prototype isa VendorPencilArray || throw(ArgumentError(
        "dist_scalar_laplacian prototype must use the input GPU vendor",
    ))
    forwarded = Base.structdiff((; kwargs...), (; prototype_θφ=prototype))
    return _stage_vendor_call(
        :dist_scalar_laplacian,
        (host_input, host_prototype) -> SHTnsKit.dist_scalar_laplacian(
            cfg, host_input;
            prototype_θφ=host_prototype, forwarded...,
        ), input, prototype,
    )
end

function SHTnsKit.dist_scalar_laplacian!(cfg::SHTnsKit.SHTConfig,
                                         output::VendorPencilArray,
                                         input::VendorPencilArray; kwargs...)
    return _stage_vendor_call(
        :dist_scalar_laplacian!,
        (host_output, host_input) -> SHTnsKit.dist_scalar_laplacian!(
            cfg, host_output, host_input; kwargs...,
        ), output, input; mutated=(1,),
    )
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, angle::Real)
    return _stage_vendor_call(:dist_SH_Zrotate, input) do host
        SHTnsKit.dist_SH_Zrotate(cfg, host, angle)
    end
end

for name in (:dist_SH_Zrotate, :dist_SH_Yrotate,
             :dist_SH_Yrotate_allgatherm!,
             :dist_SH_Yrotate_truncgatherm!)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, angle::Real,
                                  output::VendorPencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_output) -> SHTnsKit.$name(
                cfg, host_input, angle, host_output,
            ), input, output; mutated=(2,),
        )
    end
end

for name in (:dist_SH_Yrotate90, :dist_SH_Xrotate90)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray,
                                  output::VendorPencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_output) -> SHTnsKit.$name(
                cfg, host_input, host_output,
            ), input, output; mutated=(2,),
        )
    end
end

function SHTnsKit.dist_SH_rotate_euler(cfg::SHTnsKit.SHTConfig,
                                       input::VendorPencilArray,
                                       alpha::Real, beta::Real, gamma::Real,
                                       output::VendorPencilArray)
    return _stage_vendor_call(
        :dist_SH_rotate_euler,
        (host_input, host_output) -> SHTnsKit.dist_SH_rotate_euler(
            cfg, host_input, alpha, beta, gamma, host_output,
        ), input, output; mutated=(2,),
    )
end

for name in (:dist_SH_Zrotate_packed, :dist_SH_Yrotate_packed)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorArray, angle::Real;
                                  prototype_lm::VendorPencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients, angle; prototype_lm=host_prototype,
            ), coefficients, prototype_lm,
        )
    end
end

for name in (:dist_SH_Yrotate90_packed, :dist_SH_Xrotate90_packed)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorArray;
                                  prototype_lm::VendorPencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients; prototype_lm=host_prototype,
            ), coefficients, prototype_lm,
        )
    end
end

for name in (:enstrophy_l_spectrum, :enstrophy_m_spectrum)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

for name in (:energy_vector_l_spectrum, :energy_vector_m_spectrum,
             :grid_energy_vector)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::VendorPencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second) -> SHTnsKit.$name(
                cfg, host_first, host_second; kwargs...,
            ), first, second,
        )
    end
end
