module SHTns37TestCapabilities

const BACKENDS = (
    :cpu, :cuda, :amdgpu, :mpi_cpu, :mpi_cuda, :mpi_amdgpu,
)

const CAPABILITIES = (
    :scalar_real_full, :scalar_complex_full, :scalar_l, :scalar_ml,
    :scalar_batch, :packed_storage, :sphtor_full, :sphtor_l,
    :sphtor_ml, :sphtor_batch, :qst_full, :qst_l, :qst_ml, :qst_batch,
    :point, :point_complex, :latitude, :latitude_complex,
    :qst_point, :qst_latitude, :gradient_point, :operators, :rotations,
)

const TESTFILES = Dict(
    :cpu => "test/parity/runtests_cpu.jl",
    :cuda => "test/gpu/cuda/runtests.jl",
    :amdgpu => "test/gpu/amdgpu/runtests.jl",
    :mpi_cpu => "test/parity/runtests_mpi.jl",
    :mpi_cuda => "test/gpu/cuda/mpi_runtests.jl",
    :mpi_amdgpu => "test/gpu/amdgpu/mpi_runtests.jl",
)

const STORAGE_ENTRYPOINTS = (
    :nlm_calc, :nlm_cplx_calc, :LM_index, :LiM_index,
    :build_li_mi, :im_from_lm, :LM_cplx_index, :LM_cplx,
)

const TRANSFORM_ENTRYPOINTS = (
    :SHTPlan, :analysis, :synthesis, :synthesis_cplx,
    :analysis!, :synthesis!,
    :analysis_batch, :analysis_batch!,
    :synthesis_batch, :synthesis_batch!, :synthesis_batch_cplx,
    :analysis_sphtor_batch, :synthesis_sphtor_batch,
    :synthesis_sphtor_batch_cplx,
    :analysis_qst_batch, :synthesis_qst_batch, :synthesis_qst_batch_cplx,
    :analysis_sphtor!, :synthesis_sphtor!,
    :analysis_packed, :synthesis_packed,
    :analysis_packed_l, :synthesis_packed_l,
    :analysis_packed_ml, :synthesis_packed_ml,
    :analysis_packed_cplx, :synthesis_packed_cplx,
    :analysis_axisym, :synthesis_axisym,
    :analysis_axisym_l, :synthesis_axisym_l,
    :synthesis_point, :synthesis_point_cplx,
    :analysis_sphtor, :synthesis_sphtor,
    :analysis_sphtor_cplx, :synthesis_sphtor_cplx,
    :synthesis_sph, :synthesis_sph_cplx,
    :synthesis_tor, :synthesis_tor_cplx,
    :analysis_sphtor_l, :synthesis_sphtor_l,
    :synthesis_sphtor_l_cplx,
    :synthesis_sph_l, :synthesis_sph_l_cplx,
    :synthesis_tor_l, :synthesis_tor_l_cplx,
    :analysis_sphtor_ml, :synthesis_sphtor_ml,
    :synthesis_sph_ml, :synthesis_tor_ml,
    :analysis_qst, :synthesis_qst,
    :analysis_qst_cplx, :synthesis_qst_cplx,
    :analysis_qst_l, :synthesis_qst_l, :synthesis_qst_l_cplx,
    :analysis_qst_ml, :synthesis_qst_ml,
)

const LOCAL_ENTRYPOINTS = (
    :SH_to_lat, :SH_to_lat_cplx,
    :SHqst_to_lat, :SHqst_to_point, :SH_to_grad_point,
)

const OPERATOR_ENTRYPOINTS = (
    :synthesis_grad, :synthesis_grad_l, :synthesis_grad_ml,
    :divergence_from_spheroidal, :divergence_from_spheroidal!,
    :spheroidal_from_divergence, :spheroidal_from_divergence!,
    :vorticity_from_toroidal, :vorticity_from_toroidal!,
    :toroidal_from_vorticity, :toroidal_from_vorticity!,
    :mul_ct_matrix, :st_dt_matrix, :SH_mul_mx,
)

const ROTATION_ENTRYPOINTS = (
    :SH_Zrotate, :SH_Yrotate, :SH_Yrotate90, :SH_Xrotate90,
    :SHTRotation, :shtns_rotation_create, :shtns_rotation_destroy,
    :shtns_rotation_set_angles_ZYZ, :shtns_rotation_set_angles_ZXZ,
    :shtns_rotation_wigner_d_matrix, :shtns_rotation_set_angle_axis,
    :shtns_rotation_apply_cplx, :shtns_rotation_apply_real,
)

const ENTRYPOINT_GROUPS = (
    storage=STORAGE_ENTRYPOINTS,
    transforms=TRANSFORM_ENTRYPOINTS,
    local_evaluation=LOCAL_ENTRYPOINTS,
    operators=OPERATOR_ENTRYPOINTS,
    rotations=ROTATION_ENTRYPOINTS,
)

const ENTRYPOINTS = (
    STORAGE_ENTRYPOINTS...,
    TRANSFORM_ENTRYPOINTS...,
    LOCAL_ENTRYPOINTS...,
    OPERATOR_ENTRYPOINTS...,
    ROTATION_ENTRYPOINTS...,
)

end
