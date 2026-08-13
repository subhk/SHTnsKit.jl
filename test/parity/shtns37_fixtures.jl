using SHA
using Test
using TOML

isdefined(@__MODULE__, :SHTns37TestCapabilities) ||
    include(joinpath(@__DIR__, "capabilities.jl"))
using .SHTns37TestCapabilities

const SHTNS37_FIXTURE_ROOT = normpath(joinpath(@__DIR__, "..", "fixtures", "shtns37"))
const SHTNS37_MANIFEST_PATH = joinpath(SHTNS37_FIXTURE_ROOT, "manifest.toml")
const SHTNS37_GENERATOR_PATH = normpath(joinpath(@__DIR__, "..", "..", "reference",
                                                  "shtns37", "generate.c"))

_shtns37_sha256(path) = bytes2hex(open(SHA.sha256, path))

function _shtns37_read_payload(payload)
    path = joinpath(SHTNS37_FIXTURE_ROOT, payload["file"])
    bytes = read(path)
    if payload["eltype"] == "float64"
        values = reinterpret(Float64, ltoh.(reinterpret(UInt64, bytes)))
        return reshape(copy(values), Tuple(payload["shape"]))
    elseif payload["eltype"] == "complex64"
        values = reinterpret(Float64, ltoh.(reinterpret(UInt64, bytes)))
        complex_values = ComplexF64.(values[1:2:end], values[2:2:end])
        return reshape(complex_values, Tuple(payload["shape"]))
    elseif payload["eltype"] == "float32"
        values = reinterpret(Float32, ltoh.(reinterpret(UInt32, bytes)))
        return reshape(copy(values), Tuple(payload["shape"]))
    elseif payload["eltype"] == "complex32"
        values = reinterpret(Float32, ltoh.(reinterpret(UInt32, bytes)))
        complex_values = ComplexF32.(values[1:2:end], values[2:2:end])
        return reshape(complex_values, Tuple(payload["shape"]))
    end
    payload_eltype = payload["eltype"]
    error("unsupported fixture payload eltype $payload_eltype")
end

function test_shtns37_scalar_smoke()
    manifest = TOML.parsefile(SHTNS37_MANIFEST_PATH)
    fixture = only(filter(f -> f["id"] == "scalar_real_full_gauss_f64",
                          manifest["fixture"]))
    payloads = Dict(p["name"] => _shtns37_read_payload(p) for p in fixture["payload"])
    cfg = create_gauss_config(
        fixture["lmax"], fixture["nlat"];
        mmax=fixture["mmax"], mres=fixture["mres"], nlon=fixture["nphi"],
        norm=Symbol(fixture["norm"]), cs_phase=fixture["cs_phase"],
        real_norm=fixture["real_norm"],
    )
    coefficients = SHTnsKit.unpack_lm(cfg, vec(payloads["coefficients"]))
    field = payloads["field"]
    @test synthesis(CPU(), cfg, coefficients) ≈ field atol=fixture["atol"] rtol=fixture["rtol"]
    @test SHTnsKit.pack_lm(cfg, analysis(CPU(), cfg, field)) ≈
          vec(payloads["coefficients"]) atol=fixture["atol"] rtol=fixture["rtol"]
    return nothing
end

function _shtns37_config(fixture)
    kwargs = (;
        mmax=fixture["mmax"], mres=fixture["mres"], nlon=fixture["nphi"],
        norm=Symbol(fixture["norm"]), cs_phase=fixture["cs_phase"],
        real_norm=fixture["real_norm"],
    )
    grid = fixture["grid"]
    grid == "gauss" && return create_gauss_config(fixture["lmax"],fixture["nlat"];kwargs...)
    grid == "gauss_fly" && return create_gauss_fly_config(fixture["lmax"],fixture["nlat"];kwargs...)
    grid == "regular" && return create_regular_config(fixture["lmax"],fixture["nlat"];kwargs...,include_poles=false)
    grid == "regular_poles" && return create_regular_config(fixture["lmax"],fixture["nlat"];kwargs...,include_poles=true)
    error("unsupported fixture grid $grid")
end

function _shtns37_payloads(fixture)
    return Dict(p["name"] => _shtns37_read_payload(p) for p in fixture["payload"])
end

function _test_shtns37_scalar_fixture(fixture)
    cfg=_shtns37_config(fixture); p=_shtns37_payloads(fixture)
    tol=(;atol=fixture["atol"],rtol=fixture["rtol"])
    capability=Symbol(fixture["capability"])
    if capability === :scalar_real_full
        q=SHTnsKit.unpack_lm(cfg,vec(p["coefficients"])); v=p["field"]
        @test synthesis(CPU(),cfg,q) ≈ v atol=tol.atol rtol=tol.rtol
        @test SHTnsKit.pack_lm(cfg,analysis(CPU(),cfg,v)) ≈ vec(p["coefficients"]) atol=tol.atol rtol=tol.rtol
    elseif capability === :scalar_complex_full
        q=vec(p["coefficients"]); v=p["field"]
        @test synthesis_packed_cplx(CPU(),cfg,q) ≈ v atol=tol.atol rtol=tol.rtol
        @test analysis_packed_cplx(CPU(),cfg,v) ≈ q atol=tol.atol rtol=tol.rtol
    elseif capability === :scalar_l
        q=vec(p["coefficients"]); v=vec(p["field"])
        @test synthesis_packed_l(CPU(),cfg,q,fixture["ltr"]) ≈ v atol=tol.atol rtol=tol.rtol
        @test analysis_packed_l(CPU(),cfg,v,fixture["ltr"]) ≈
              map(eachindex(q)) do i
                  cfg.li[i] <= fixture["ltr"] ? q[i] : zero(eltype(q))
              end atol=tol.atol rtol=tol.rtol
    elseif capability === :scalar_ml
        q=vec(p["coefficients"]); v=vec(p["field"]); im=fixture["stored_im"]
        scale=fixture["fixed_mode_scale"]
        @test synthesis_packed_ml(CPU(),cfg,im,q,fixture["ltr"]) ≈ scale .* v atol=tol.atol rtol=tol.rtol
        @test analysis_packed_ml(CPU(),cfg,im,scale .* v,fixture["ltr"]) ≈ q atol=tol.atol rtol=tol.rtol
    elseif capability === :scalar_batch
        q=p["coefficients"]; v=p["field"]
        dense=cat((SHTnsKit.unpack_lm(cfg,q[:,k]) for k in axes(q,2))...;dims=3)
        @test synthesis_batch(CPU(),cfg,dense) ≈ v atol=tol.atol rtol=tol.rtol
        got=analysis_batch(CPU(),cfg,v)
        for k in axes(q,2)
            @test SHTnsKit.pack_lm(cfg,got[:,:,k]) ≈ q[:,k] atol=tol.atol rtol=tol.rtol
        end
    elseif capability === :packed_storage
        q=vec(p["coefficients"]); v=vec(p["field"])
        @test synthesis_packed(CPU(),cfg,q) ≈ v atol=tol.atol rtol=tol.rtol
        @test analysis_packed(CPU(),cfg,v) ≈ q atol=tol.atol rtol=tol.rtol
    else
        return false
    end
    return true
end

function test_shtns37_scalar_fixtures()
    manifest=TOML.parsefile(SHTNS37_MANIFEST_PATH)
    @testset "SHTns 3.7 scalar fixtures" begin
        for fixture in manifest["fixture"]
            Symbol(fixture["capability"]) in (:scalar_real_full,:scalar_complex_full,
                :scalar_l,:scalar_ml,:scalar_batch,:packed_storage) || continue
            fixture_id = fixture["id"]
            @testset "$fixture_id" begin
                @test _test_shtns37_scalar_fixture(fixture)
            end
        end
    end
    return nothing
end

_shtns37_dense(cfg,q)=SHTnsKit.unpack_lm(cfg,vec(q))
_shtns37_batch_dense(cfg,p,n)=cat((_shtns37_dense(cfg,p[n][:,k]) for k in axes(p[n],2))...;dims=3)
function _shtns37_truncated_packed(cfg,q,ltr)
    return map(eachindex(q)) do i
        cfg.li[i] <= ltr ? q[i] : zero(eltype(q))
    end
end

function _test_shtns37_vector_fixture(fixture)
    cfg=_shtns37_config(fixture);p=_shtns37_payloads(fixture);cap=Symbol(fixture["capability"])
    atol=fixture["atol"];rtol=fixture["rtol"]
    if cap === :sphtor_full
        S=_shtns37_dense(cfg,p["S"]);T=_shtns37_dense(cfg,p["T"])
        got=synthesis_sphtor(CPU(),cfg,S,T)
        @test got[1] ≈ p["Vt"] atol=atol rtol=rtol; @test got[2] ≈ p["Vp"] atol=atol rtol=rtol
        back=analysis_sphtor(CPU(),cfg,p["Vt"],p["Vp"])
        @test SHTnsKit.pack_lm(cfg,back[1]) ≈ vec(p["S"]) atol=atol rtol=rtol
        @test SHTnsKit.pack_lm(cfg,back[2]) ≈ vec(p["T"]) atol=atol rtol=rtol
    elseif cap === :sphtor_l
        S=_shtns37_dense(cfg,p["S"]);T=_shtns37_dense(cfg,p["T"]);ltr=fixture["ltr"]
        got=synthesis_sphtor_l(CPU(),cfg,S,T,ltr)
        @test got[1] ≈ p["Vt"] atol=atol rtol=rtol; @test got[2] ≈ p["Vp"] atol=atol rtol=rtol
        back=analysis_sphtor_l(CPU(),cfg,p["Vt"],p["Vp"],ltr)
        @test SHTnsKit.pack_lm(cfg,back[1]) ≈ _shtns37_truncated_packed(cfg,vec(p["S"]),ltr) atol=atol rtol=rtol
        @test SHTnsKit.pack_lm(cfg,back[2]) ≈ _shtns37_truncated_packed(cfg,vec(p["T"]),ltr) atol=atol rtol=rtol
    elseif cap === :sphtor_ml
        scale=fixture["fixed_mode_scale"];im=fixture["stored_im"]
        got=synthesis_sphtor_ml(CPU(),cfg,im,vec(p["S"]),vec(p["T"]),fixture["ltr"])
        @test got[1] ≈ scale .* vec(p["Vt"]) atol=atol rtol=rtol; @test got[2] ≈ scale .* vec(p["Vp"]) atol=atol rtol=rtol
        back=analysis_sphtor_ml(CPU(),cfg,im,scale .* vec(p["Vt"]),scale .* vec(p["Vp"]),fixture["ltr"])
        @test back[1] ≈ vec(p["S"]) atol=atol rtol=rtol; @test back[2] ≈ vec(p["T"]) atol=atol rtol=rtol
    elseif cap === :sphtor_batch
        S=cat((_shtns37_dense(cfg,p["S"][:,k]) for k in axes(p["S"],2))...;dims=3)
        T=cat((_shtns37_dense(cfg,p["T"][:,k]) for k in axes(p["T"],2))...;dims=3)
        got=synthesis_sphtor_batch(CPU(),cfg,S,T)
        @test got[1] ≈ p["Vt"] atol=atol rtol=rtol; @test got[2] ≈ p["Vp"] atol=atol rtol=rtol
        back=analysis_sphtor_batch(CPU(),cfg,p["Vt"],p["Vp"])
        for k in axes(p["S"],2)
            @test SHTnsKit.pack_lm(cfg,back[1][:,:,k]) ≈ p["S"][:,k] atol=atol rtol=rtol
            @test SHTnsKit.pack_lm(cfg,back[2][:,:,k]) ≈ p["T"][:,k] atol=atol rtol=rtol
        end
    else
        return false
    end
    return true
end

function test_shtns37_vector_fixtures()
    manifest=TOML.parsefile(SHTNS37_MANIFEST_PATH)
    @testset "SHTns 3.7 vector fixtures" begin
        for fixture in manifest["fixture"]
            Symbol(fixture["capability"]) in (:sphtor_full,:sphtor_l,:sphtor_ml,:sphtor_batch) || continue
            fixture_id=fixture["id"]
            @testset "$fixture_id" begin @test _test_shtns37_vector_fixture(fixture) end
        end
    end
    return nothing
end

function _test_shtns37_qst_fixture(f)
    cfg=_shtns37_config(f);p=_shtns37_payloads(f);cap=Symbol(f["capability"]);a=f["atol"];r=f["rtol"]
    if cap in (:qst_full,:qst_l)
        Q=_shtns37_dense(cfg,p["Q"]);S=_shtns37_dense(cfg,p["S"]);T=_shtns37_dense(cfg,p["T"])
        got=cap===:qst_full ? synthesis_qst(CPU(),cfg,Q,S,T) : synthesis_qst_l(CPU(),cfg,Q,S,T,f["ltr"])
        @test got[1] ≈ p["Vr"] atol=a rtol=r;@test got[2] ≈ p["Vt"] atol=a rtol=r;@test got[3] ≈ p["Vp"] atol=a rtol=r
        back=cap===:qst_full ? analysis_qst(CPU(),cfg,p["Vr"],p["Vt"],p["Vp"]) : analysis_qst_l(CPU(),cfg,p["Vr"],p["Vt"],p["Vp"],f["ltr"])
        for (i,n) in enumerate(("Q","S","T"))
            expected=cap===:qst_l ? _shtns37_truncated_packed(cfg,vec(p[n]),f["ltr"]) : vec(p[n])
            @test SHTnsKit.pack_lm(cfg,back[i]) ≈ expected atol=a rtol=r
        end
    elseif cap===:qst_ml
        sc=f["fixed_mode_scale"];im=f["stored_im"]
        got=synthesis_qst_ml(CPU(),cfg,im,vec(p["Q"]),vec(p["S"]),vec(p["T"]),f["ltr"])
        for (i,n) in enumerate(("Vr","Vt","Vp"));@test got[i] ≈ sc.*vec(p[n]) atol=a rtol=r;end
        back=analysis_qst_ml(CPU(),cfg,im,sc.*vec(p["Vr"]),sc.*vec(p["Vt"]),sc.*vec(p["Vp"]),f["ltr"])
        for (i,n) in enumerate(("Q","S","T"));@test back[i] ≈ vec(p[n]) atol=a rtol=r;end
    elseif cap===:qst_batch
        dense(n)=cat((_shtns37_dense(cfg,p[n][:,k]) for k in axes(p[n],2))...;dims=3)
        got=synthesis_qst_batch(CPU(),cfg,dense("Q"),dense("S"),dense("T"))
        for (i,n) in enumerate(("Vr","Vt","Vp"));@test got[i] ≈ p[n] atol=a rtol=r;end
        back=analysis_qst_batch(CPU(),cfg,p["Vr"],p["Vt"],p["Vp"])
        for k in axes(p["Q"],2),(i,n) in enumerate(("Q","S","T"));@test SHTnsKit.pack_lm(cfg,back[i][:,:,k]) ≈ p[n][:,k] atol=a rtol=r;end
    else
        return false
    end
    true
end

function test_shtns37_qst_fixtures()
    manifest=TOML.parsefile(SHTNS37_MANIFEST_PATH)
    @testset "SHTns 3.7 QST fixtures" begin
        for f in manifest["fixture"]
            Symbol(f["capability"]) in (:qst_full,:qst_l,:qst_ml,:qst_batch)||continue
            id=f["id"];@testset "$id" begin @test _test_shtns37_qst_fixture(f) end
        end
    end
end

function _test_shtns37_local_fixture(f)
    cfg=_shtns37_config(f);p=_shtns37_payloads(f);cap=Symbol(f["capability"]);a=f["atol"];r=f["rtol"]
    if cap===:point
        @test synthesis_point(CPU(),cfg,_shtns37_dense(cfg,p["Q"]),f["cost"],f["phi"]) ≈ p["value"][1] atol=a rtol=r
    elseif cap===:point_complex
        @test synthesis_point_cplx(CPU(),cfg,vec(p["A"]),f["cost"],f["phi"]) ≈ p["value"][1] atol=a rtol=r
    elseif cap===:latitude
        @test SH_to_lat(CPU(),cfg,vec(p["Q"]),f["cost"];nphi=f["nphi"],ltr=f["ltr"],mtr=f["mmax"]) ≈ vec(p["values"]) atol=a rtol=r
    elseif cap===:latitude_complex
        @test SH_to_lat_cplx(CPU(),cfg,vec(p["A"]),f["cost"];nphi=f["nphi"],ltr=f["ltr"]) ≈ vec(p["values"]) atol=a rtol=r
    elseif cap===:qst_point
        got=SHqst_to_point(CPU(),cfg,vec(p["Q"]),vec(p["S"]),vec(p["T"]),f["cost"],f["phi"])
        @test collect(got) ≈ vec(p["value"]) atol=a rtol=r
    elseif cap===:qst_latitude
        got=SHqst_to_lat(CPU(),cfg,vec(p["Q"]),vec(p["S"]),vec(p["T"]),f["cost"];nphi=f["nphi"],ltr=f["ltr"],mtr=f["mmax"])
        for (i,n) in enumerate(("Vr","Vt","Vp"));@test got[i] ≈ vec(p[n]) atol=a rtol=r;end
    elseif cap===:gradient_point
        got=SH_to_grad_point(CPU(),cfg,vec(p["Dr"]),vec(p["S"]),f["cost"],f["phi"])
        @test collect(got) ≈ vec(p["value"]) atol=a rtol=r
    else
        return false
    end
    true
end
function test_shtns37_local_fixtures()
    manifest=TOML.parsefile(SHTNS37_MANIFEST_PATH)
    @testset "SHTns 3.7 local fixtures" begin
        for f in manifest["fixture"]
            Symbol(f["capability"]) in (:point,:point_complex,:latitude,:latitude_complex,:qst_point,:qst_latitude,:gradient_point)||continue
            id=f["id"];@testset "$id" begin @test _test_shtns37_local_fixture(f) end
        end
    end
end

function test_shtns37_operator_rotation_fixtures()
    manifest=TOML.parsefile(SHTNS37_MANIFEST_PATH)
    @testset "SHTns 3.7 operator and rotation fixtures" begin
        op=only(filter(f->f["capability"]=="operators",manifest["fixture"]));cfg=_shtns37_config(op);p=_shtns37_payloads(op);a=op["atol"];r=op["rtol"]
        ct=zeros(Float64,2cfg.nlm);dt=similar(ct);mul_ct_matrix(CPU(),cfg,ct);st_dt_matrix(CPU(),cfg,dt)
        @test ct ≈ vec(p["ct_matrix"]) atol=a rtol=r;@test dt ≈ vec(p["dt_matrix"]) atol=a rtol=r
        rct=zeros(ComplexF64,cfg.nlm);rdt=similar(rct);SH_mul_mx(CPU(),cfg,ct,vec(p["Q"]),rct);SH_mul_mx(CPU(),cfg,dt,vec(p["Q"]),rdt)
        @test rct ≈ vec(p["ct_result"]) atol=a rtol=r;@test rdt ≈ vec(p["dt_result"]) atol=a rtol=r
        rot=only(filter(f->f["capability"]=="rotations",manifest["fixture"]));cfg=_shtns37_config(rot);p=_shtns37_payloads(rot);a=rot["atol"];r=rot["rtol"]
        z=similar(vec(p["Q"]));y=similar(z);SH_Zrotate(CPU(),cfg,vec(p["Q"]),rot["z_angle"],z);SH_Yrotate(CPU(),cfg,vec(p["Q"]),rot["y_angle"],y)
        @test z ≈ vec(p["Z"]) atol=a rtol=r;@test y ≈ vec(p["Y"]) atol=a rtol=r
    end
end

function test_shtns37_fixtures_cpu()
    test_shtns37_fixture_manifest()
    test_shtns37_scalar_fixtures()
    test_shtns37_vector_fixtures()
    test_shtns37_qst_fixtures()
    test_shtns37_local_fixtures()
    test_shtns37_operator_rotation_fixtures()
    return nothing
end

_shtns37_host(x::Tuple)=map(_shtns37_host,x)
_shtns37_host(x)=Array(x)

"""Run every generated oracle through one functional vendor GPU backend."""
function test_shtns37_gpu_fixtures(to_device)
    manifest=TOML.parsefile(SHTNS37_MANIFEST_PATH)
    @testset "SHTns 3.7 GPU fixtures" begin
        for f in manifest["fixture"]
            cfg=_shtns37_config(f);p=_shtns37_payloads(f);cap=Symbol(f["capability"]);a=f["atol"];r=f["rtol"]
            id=f["id"]
            @testset "$id" begin
                if cap===:scalar_real_full
                    got=synthesis(GPU(),cfg,to_device(_shtns37_dense(cfg,p["coefficients"])))
                    @test Array(got) ≈ p["field"] atol=a rtol=r
                elseif cap===:scalar_complex_full
                    @test Array(synthesis_packed_cplx(cfg,to_device(vec(p["coefficients"])))) ≈ p["field"] atol=a rtol=r
                elseif cap in (:scalar_l,:packed_storage)
                    got=cap===:scalar_l ? synthesis_packed_l(cfg,to_device(vec(p["coefficients"])),f["ltr"]) : synthesis_packed(cfg,to_device(vec(p["coefficients"])))
                    @test Array(got) ≈ vec(p["field"]) atol=a rtol=r
                elseif cap===:scalar_ml
                    got=synthesis_packed_ml(cfg,f["stored_im"],to_device(vec(p["coefficients"])),f["ltr"])
                    @test Array(got) ≈ f["fixed_mode_scale"].*vec(p["field"]) atol=a rtol=r
                elseif cap===:scalar_batch
                    q=cat((_shtns37_dense(cfg,p["coefficients"][:,k]) for k in axes(p["coefficients"],2))...;dims=3)
                    @test Array(synthesis_batch(cfg,to_device(q))) ≈ p["field"] atol=a rtol=r
                elseif cap in (:sphtor_full,:sphtor_l)
                    S=to_device(_shtns37_dense(cfg,p["S"]));T=to_device(_shtns37_dense(cfg,p["T"]));got=cap===:sphtor_full ? synthesis_sphtor(GPU(),cfg,S,T) : synthesis_sphtor_l(GPU(),cfg,S,T,f["ltr"])
                    @test Array(got[1]) ≈ p["Vt"] atol=a rtol=r;@test Array(got[2]) ≈ p["Vp"] atol=a rtol=r
                elseif cap===:sphtor_ml
                    got=synthesis_sphtor_ml(GPU(),cfg,f["stored_im"],to_device(vec(p["S"])),to_device(vec(p["T"])),f["ltr"]);sc=f["fixed_mode_scale"]
                    @test Array(got[1]) ≈ sc.*vec(p["Vt"]) atol=a rtol=r;@test Array(got[2]) ≈ sc.*vec(p["Vp"]) atol=a rtol=r
                elseif cap===:sphtor_batch
                    Sbatch=cat((_shtns37_dense(cfg,p["S"][:,k]) for k in axes(p["S"],2))...;dims=3)
                    Tbatch=cat((_shtns37_dense(cfg,p["T"][:,k]) for k in axes(p["T"],2))...;dims=3)
                    got=synthesis_sphtor_batch(GPU(),cfg,to_device(Sbatch),to_device(Tbatch))
                    @test Array(got[1]) ≈ p["Vt"] atol=a rtol=r;@test Array(got[2]) ≈ p["Vp"] atol=a rtol=r
                elseif cap in (:qst_full,:qst_l)
                    Q=to_device(_shtns37_dense(cfg,p["Q"]));S=to_device(_shtns37_dense(cfg,p["S"]));T=to_device(_shtns37_dense(cfg,p["T"]));got=cap===:qst_full ? synthesis_qst(GPU(),cfg,Q,S,T) : synthesis_qst_l(GPU(),cfg,Q,S,T,f["ltr"])
                    for (i,n) in enumerate(("Vr","Vt","Vp"));@test Array(got[i]) ≈ p[n] atol=a rtol=r;end
                elseif cap===:qst_ml
                    got=synthesis_qst_ml(GPU(),cfg,f["stored_im"],to_device(vec(p["Q"])),to_device(vec(p["S"])),to_device(vec(p["T"])),f["ltr"]);sc=f["fixed_mode_scale"]
                    for (i,n) in enumerate(("Vr","Vt","Vp"));@test Array(got[i]) ≈ sc.*vec(p[n]) atol=a rtol=r;end
                elseif cap===:qst_batch
                    Qbatch=cat((_shtns37_dense(cfg,p["Q"][:,k]) for k in axes(p["Q"],2))...;dims=3)
                    Sbatch=cat((_shtns37_dense(cfg,p["S"][:,k]) for k in axes(p["S"],2))...;dims=3)
                    Tbatch=cat((_shtns37_dense(cfg,p["T"][:,k]) for k in axes(p["T"],2))...;dims=3)
                    got=synthesis_qst_batch(GPU(),cfg,to_device(Qbatch),to_device(Sbatch),to_device(Tbatch))
                    for (i,n) in enumerate(("Vr","Vt","Vp"));@test Array(got[i]) ≈ p[n] atol=a rtol=r;end
                elseif cap===:point
                    @test synthesis_point(GPU(),cfg,to_device(_shtns37_dense(cfg,p["Q"])),f["cost"],f["phi"]) ≈ p["value"][1] atol=a rtol=r
                elseif cap===:point_complex
                    @test synthesis_point_cplx(GPU(),cfg,to_device(vec(p["A"])),f["cost"],f["phi"]) ≈ p["value"][1] atol=a rtol=r
                elseif cap===:latitude
                    @test Array(SH_to_lat(GPU(),cfg,to_device(vec(p["Q"])),f["cost"];nphi=f["nphi"],ltr=f["ltr"],mtr=f["mmax"])) ≈ vec(p["values"]) atol=a rtol=r
                elseif cap===:latitude_complex
                    @test Array(SH_to_lat_cplx(GPU(),cfg,to_device(vec(p["A"])),f["cost"];nphi=f["nphi"],ltr=f["ltr"])) ≈ vec(p["values"]) atol=a rtol=r
                elseif cap in (:qst_point,:qst_latitude)
                    Q=to_device(vec(p["Q"]));S=to_device(vec(p["S"]));T=to_device(vec(p["T"]));got=cap===:qst_point ? SHqst_to_point(GPU(),cfg,Q,S,T,f["cost"],f["phi"]) : SHqst_to_lat(GPU(),cfg,Q,S,T,f["cost"];nphi=f["nphi"],ltr=f["ltr"],mtr=f["mmax"])
                    if cap===:qst_point
                        @test collect(got) ≈ vec(p["value"]) atol=a rtol=r
                    else
                        for (i,n) in enumerate(("Vr","Vt","Vp"));@test Array(got[i]) ≈ vec(p[n]) atol=a rtol=r;end
                    end
                elseif cap===:gradient_point
                    got=SH_to_grad_point(GPU(),cfg,to_device(vec(p["Dr"])),to_device(vec(p["S"])),f["cost"],f["phi"]);@test collect(got) ≈ vec(p["value"]) atol=a rtol=r
                elseif cap===:operators
                    ct=to_device(zeros(Float64,2cfg.nlm));dt=to_device(zeros(Float64,2cfg.nlm));mul_ct_matrix(GPU(),cfg,ct);st_dt_matrix(GPU(),cfg,dt)
                    @test Array(ct) ≈ vec(p["ct_matrix"]) atol=a rtol=r;@test Array(dt) ≈ vec(p["dt_matrix"]) atol=a rtol=r
                    rct=to_device(zeros(ComplexF64,cfg.nlm));rdt=to_device(zeros(ComplexF64,cfg.nlm));Q=to_device(vec(p["Q"]));SH_mul_mx(GPU(),cfg,ct,Q,rct);SH_mul_mx(GPU(),cfg,dt,Q,rdt)
                    @test Array(rct) ≈ vec(p["ct_result"]) atol=a rtol=r;@test Array(rdt) ≈ vec(p["dt_result"]) atol=a rtol=r
                elseif cap===:rotations
                    z=to_device(zeros(ComplexF64,cfg.nlm));y=to_device(zeros(ComplexF64,cfg.nlm));Q=to_device(vec(p["Q"]));SH_Zrotate(GPU(),cfg,Q,f["z_angle"],z);SH_Yrotate(GPU(),cfg,Q,f["y_angle"],y)
                    @test Array(z) ≈ vec(p["Z"]) atol=a rtol=r;@test Array(y) ≈ vec(p["Y"]) atol=a rtol=r
                end
            end
        end
    end
end

"""Run every generated oracle through the MPI/PencilArray backend."""
function _shtns37_place_scalar_batch(cfg, values, kind, comm)
    pen = kind === :spatial ? create_spatial_pencil(cfg; comm) :
                              create_spectral_pencil(cfg; comm)
    result = PencilArray{eltype(values)}(undef, pen, size(values, 3))
    ranges = PencilArrays.range_local(pen)
    @inbounds for k in axes(values, 3),
                  (j, jg) in pairs(ranges[2]), (i, ig) in pairs(ranges[1])
        parent(result)[i, j, k] = values[ig, jg, k]
    end
    return result
end

function test_shtns37_mpi_fixtures(comm)
    manifest=TOML.parsefile(SHTNS37_MANIFEST_PATH)
    @testset "SHTns 3.7 MPI fixtures" begin
        for f in manifest["fixture"]
            cfg=_shtns37_config(f);p=_shtns37_payloads(f);cap=Symbol(f["capability"]);a=f["atol"];r=f["rtol"]
            RT=f["precision"] == "float32" ? Float32 : Float64
            prototype=place(MPIScalarAdapter(comm),cfg,zeros(RT,cfg.nlat,cfg.nlon),:spatial);id=f["id"]
            @testset "$id" begin
                if cap===:scalar_real_full
                    q=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["coefficients"]);comm);got=synthesis(cfg,q;prototype_θφ=prototype);@test _collect_spatial(got,cfg) ≈ p["field"] atol=a rtol=r
                elseif cap===:scalar_complex_full
                    q=_place_distributed_vector(vec(p["coefficients"]),comm);cp=place(MPIScalarAdapter(comm),cfg,zeros(ComplexF64,cfg.nlat,cfg.nlon),:spatial);got=synthesis_packed_cplx(cfg,q;prototype_θφ=cp);@test _collect_spatial(got,cfg) ≈ p["field"] atol=a rtol=r
                elseif cap in (:scalar_l,:packed_storage)
                    q=_place_distributed_vector(vec(p["coefficients"]),comm);got=cap===:scalar_l ? synthesis_packed_l(cfg,q,f["ltr"];prototype_θφ=prototype) : synthesis_packed(cfg,q;prototype_θφ=prototype);@test vec(_collect_spatial(got,cfg)) ≈ vec(p["field"]) atol=a rtol=r
                elseif cap===:scalar_ml
                    q=_place_distributed_vector(vec(p["coefficients"]),comm);got=synthesis_packed_ml(cfg,f["stored_im"],q,f["ltr"]);@test _collect_distributed_vector(got) ≈ f["fixed_mode_scale"].*vec(p["field"]) atol=a rtol=r
                elseif cap===:scalar_batch
                    q=cat((_shtns37_dense(cfg,p["coefficients"][:,k]) for k in axes(p["coefficients"],2))...;dims=3);dq=_shtns37_place_scalar_batch(cfg,q,:spectral,comm);proto=_shtns37_place_scalar_batch(cfg,zeros(eltype(p["field"]),size(p["field"])),:spatial,comm);got=synthesis_batch(cfg,dq;prototype_θφ=proto);@test _collect_distributed_batch(got,comm) ≈ p["field"] atol=a rtol=r
                elseif cap in (:sphtor_full,:sphtor_l)
                    S=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["S"]);comm);T=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["T"]);comm);got=cap===:sphtor_full ? synthesis_sphtor(cfg,S,T;prototype_θφ=prototype) : synthesis_sphtor_l(cfg,S,T,f["ltr"];prototype_θφ=prototype);@test _collect_spatial(got[1],cfg) ≈ p["Vt"] atol=a rtol=r;@test _collect_spatial(got[2],cfg) ≈ p["Vp"] atol=a rtol=r
                elseif cap===:sphtor_ml
                    S=_place_distributed_vector(vec(p["S"]),comm);T=_place_distributed_vector(vec(p["T"]),comm);got=synthesis_sphtor_ml(cfg,f["stored_im"],S,T,f["ltr"]);sc=f["fixed_mode_scale"];@test _collect_distributed_vector(got[1]) ≈ sc.*vec(p["Vt"]) atol=a rtol=r;@test _collect_distributed_vector(got[2]) ≈ sc.*vec(p["Vp"]) atol=a rtol=r
                elseif cap===:sphtor_batch
                    S=_place_distributed_batch(cfg,_shtns37_batch_dense(cfg,p,"S"),:spectral,comm);T=_place_distributed_batch(cfg,_shtns37_batch_dense(cfg,p,"T"),:spectral,comm);got=synthesis_sphtor_batch(cfg,S,T);@test _collect_distributed_batch(got[1],comm) ≈ p["Vt"] atol=a rtol=r;@test _collect_distributed_batch(got[2],comm) ≈ p["Vp"] atol=a rtol=r
                elseif cap in (:qst_full,:qst_l)
                    Q=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["Q"]);comm);S=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["S"]);comm);T=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["T"]);comm);got=cap===:qst_full ? synthesis_qst(cfg,Q,S,T;prototype_θφ=prototype) : synthesis_qst_l(cfg,Q,S,T,f["ltr"];prototype_θφ=prototype);for (i,n) in enumerate(("Vr","Vt","Vp"));@test _collect_spatial(got[i],cfg) ≈ p[n] atol=a rtol=r;end
                elseif cap===:qst_ml
                    Q=_place_distributed_vector(vec(p["Q"]),comm);S=_place_distributed_vector(vec(p["S"]),comm);T=_place_distributed_vector(vec(p["T"]),comm);got=synthesis_qst_ml(cfg,f["stored_im"],Q,S,T,f["ltr"]);sc=f["fixed_mode_scale"];for (i,n) in enumerate(("Vr","Vt","Vp"));@test _collect_distributed_vector(got[i]) ≈ sc.*vec(p[n]) atol=a rtol=r;end
                elseif cap===:qst_batch
                    Q=_place_distributed_batch(cfg,_shtns37_batch_dense(cfg,p,"Q"),:spectral,comm);S=_place_distributed_batch(cfg,_shtns37_batch_dense(cfg,p,"S"),:spectral,comm);T=_place_distributed_batch(cfg,_shtns37_batch_dense(cfg,p,"T"),:spectral,comm);got=synthesis_qst_batch(cfg,Q,S,T);for (i,n) in enumerate(("Vr","Vt","Vp"));@test _collect_distributed_batch(got[i],comm) ≈ p[n] atol=a rtol=r;end
                elseif cap in (:point,:point_complex,:latitude,:latitude_complex,:qst_point,:qst_latitude,:gradient_point)
                    if cap===:point
                        Q=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["Q"]);comm);@test synthesis_point(cfg,Q,f["cost"],f["phi"]) ≈ p["value"][1] atol=a rtol=r
                    elseif cap===:point_complex
                        A=_place_distributed_vector(vec(p["A"]),comm);@test synthesis_point_cplx(cfg,A,f["cost"],f["phi"]) ≈ p["value"][1] atol=a rtol=r
                    elseif cap===:latitude
                        Q=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["Q"]);comm);@test SH_to_lat(cfg,Q,f["cost"];nphi=f["nphi"],ltr=f["ltr"],mtr=f["mmax"]) ≈ vec(p["values"]) atol=a rtol=r
                    elseif cap===:latitude_complex
                        A=_place_distributed_vector(vec(p["A"]),comm);@test SH_to_lat_cplx(cfg,A,f["cost"];nphi=f["nphi"],ltr=f["ltr"]) ≈ vec(p["values"]) atol=a rtol=r
                    else
                        names=cap===:gradient_point ? ("Dr","S") : ("Q","S","T");arrays=map(n->matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p[n]);comm),names)
                        got=cap===:gradient_point ? SH_to_grad_point(cfg,arrays...,f["cost"],f["phi"]) : cap===:qst_point ? SHqst_to_point(cfg,arrays...,f["cost"],f["phi"]) : SHqst_to_lat(cfg,arrays...,f["cost"];nphi=f["nphi"],ltr=f["ltr"],mtr=f["mmax"])
                        if cap===:qst_latitude;for (i,n) in enumerate(("Vr","Vt","Vp"));@test got[i] ≈ vec(p[n]) atol=a rtol=r;end;else;@test collect(got) ≈ vec(p["value"]) atol=a rtol=r;end
                    end
                elseif cap===:operators
                    Q=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["Q"]);comm);rct=similar(Q);rdt=similar(Q);SH_mul_mx(CPU(),cfg,vec(p["ct_matrix"]),Q,rct);SH_mul_mx(CPU(),cfg,vec(p["dt_matrix"]),Q,rdt)
                    @test SHTnsKit.pack_lm(cfg,spectral_pencil_to_matrix(cfg,rct)) ≈ vec(p["ct_result"]) atol=a rtol=r;@test SHTnsKit.pack_lm(cfg,spectral_pencil_to_matrix(cfg,rdt)) ≈ vec(p["dt_result"]) atol=a rtol=r
                elseif cap===:rotations
                    Q=matrix_to_spectral_pencil(cfg,_shtns37_dense(cfg,p["Q"]);comm);z=similar(Q);y=similar(Q);SHTnsKit.dist_SH_Zrotate(cfg,Q,f["z_angle"],z);SHTnsKit.dist_SH_Yrotate(cfg,Q,f["y_angle"],y)
                    @test SHTnsKit.pack_lm(cfg,spectral_pencil_to_matrix(cfg,z)) ≈ vec(p["Z"]) atol=a rtol=r;@test SHTnsKit.pack_lm(cfg,spectral_pencil_to_matrix(cfg,y)) ≈ vec(p["Y"]) atol=a rtol=r
                end
            end
        end
    end
end

function test_shtns37_fixture_manifest()
    @testset "SHTns 3.7 fixture manifest" begin
        @test isfile(SHTNS37_MANIFEST_PATH)
        @test isfile(SHTNS37_GENERATOR_PATH)
        (isfile(SHTNS37_MANIFEST_PATH) && isfile(SHTNS37_GENERATOR_PATH)) || return

        manifest = TOML.parsefile(SHTNS37_MANIFEST_PATH)
        @test manifest["format_version"] == 1
        @test manifest["shtns_version"] == "3.7"
        @test manifest["shtns_interface"] == 0x307A0
        @test manifest["upstream_tag"] == "v3.7"
        @test manifest["upstream_commit"] ==
              "4e04fba84ea156974df5edaf4ee856c0f4f86e77"
        @test manifest["upstream_archive_sha256"] ==
              "5c6a2d585211232a030c6fbbb08f6a794dd1aab987d31511ef53deea12138d97"
        @test manifest["generator_source_sha256"] == _shtns37_sha256(SHTNS37_GENERATOR_PATH)

        fixtures = manifest["fixture"]
        @test !isempty(fixtures)
        @test Set(Symbol(f["capability"]) for f in fixtures) ==
              Set(SHTns37TestCapabilities.CAPABILITIES)
        @test Set(f["grid"] for f in fixtures) ==
              Set(("gauss", "gauss_fly", "regular", "regular_poles"))
        @test Set(f["norm"] for f in fixtures) ==
              Set(("orthonormal", "fourpi", "schmidt"))
        @test Set(f["cs_phase"] for f in fixtures) == Set((false, true))
        @test Set(f["real_norm"] for f in fixtures) == Set((false, true))
        @test Set(f["precision"] for f in fixtures) == Set(("float32", "float64"))
        @test Set(f["mres"] for f in fixtures) == Set((1, 2))
        @test any(f -> f["ltr"] < f["lmax"], fixtures)

        ids = [f["id"] for f in fixtures]
        @test length(ids) == length(unique(ids))
        referenced_payloads = Set(
            payload["file"] for fixture in fixtures for payload in fixture["payload"]
        )
        @test referenced_payloads == Set(
            filter(name -> endswith(name, ".bin"), readdir(SHTNS37_FIXTURE_ROOT))
        )
        for fixture in fixtures
            @test fixture["lmax"] >= fixture["mmax"] >= 0
            @test fixture["mres"] in (1, 2)
            @test 0 <= fixture["ltr"] <= fixture["lmax"]
            @test fixture["nlat"] > fixture["lmax"]
            @test fixture["nphi"] >= 2fixture["mmax"] + 1
            @test fixture["atol"] > 0
            @test fixture["rtol"] > 0
            if fixture["precision"] == "float32"
                @test fixture["precision_provenance"] ==
                      "little-endian Float32 downcast of independent SHTns 3.7 FP64 oracle"
            end
            @test !isempty(fixture["payload"])
            for payload in fixture["payload"]
                path = joinpath(SHTNS37_FIXTURE_ROOT, payload["file"])
                @test payload["endian"] == "little"
                @test payload["eltype"] in ("float32", "float64", "complex32", "complex64")
                @test all(>(0), payload["shape"])
                @test isfile(path)
                isfile(path) || continue
                @test filesize(path) == payload["bytes"]
                @test _shtns37_sha256(path) == payload["sha256"]
            end
        end
    end
    return nothing
end
