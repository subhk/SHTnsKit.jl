module CompatibilityInventory

using SHA
using SHTnsKit

export inventory_sources, method_compatible, method_fingerprint,
       method_from_fixture, method_to_fixture, resolve_type_expression,
       source_pairs, revision_source_pairs

"""Return sorted `relative_path => source` pairs for every Julia file below `src`."""
function source_pairs(root::AbstractString)
    src = joinpath(root, "src")
    files = String[]
    for (dir, _, names) in walkdir(src), name in names
        endswith(name, ".jl") && push!(files, joinpath(dir, name))
    end
    sort!(files)
    return [replace(relpath(path, root), '\\' => '/') => read(path, String)
            for path in files]
end

"Return sorted Julia sources below `src` exactly as stored in `revision`."
function revision_source_pairs(root::AbstractString, revision::AbstractString)
    listing = read(`git -C $root ls-tree -r --name-only $revision -- src`, String)
    files = sort!(filter(path -> endswith(path, ".jl"), split(listing)))
    return [path => read(`git -C $root show $revision:$path`, String) for path in files]
end

_without_lines(ex) = ex
function _without_lines(ex::Expr)
    copy = deepcopy(ex)
    Base.remove_linenums!(copy)
    return copy
end

function _syntax(ex)
    ex === nothing && return ""
    ex isa Symbol && return String(ex)
    ex isa QuoteNode && return ":" * String(ex.value)
    ex isa Expr || return repr(ex)
    return sprint(Base.show_unquoted, _without_lines(ex); context=:limit => false)
end

function _binding_name(ex)
    ex isa Symbol && return String(ex)
    ex isa QuoteNode && return _binding_name(ex.value)
    ex isa GlobalRef && return String(ex.name)
    if ex isa Expr
        ex.head in (:call, :<:) && return _binding_name(ex.args[1])
        ex.head === :. && return _binding_name(ex.args[end])
        ex.head === :curly && return _binding_name(ex.args[1])
        ex.head === :where && return _binding_name(ex.args[1])
        ex.head === :(::) && return _binding_name(ex.args[1])
    end
    return nothing
end

function _argument(arg; keyword::Bool=false)
    default = nothing
    kind = keyword ? "keyword" : "required"
    if arg isa Expr && arg.head in (:kw, :(=))
        arg, default = arg.args
        kind = keyword ? "keyword" : "optional"
    end
    vararg = arg isa Expr && arg.head === :...
    if vararg
        arg = arg.args[1]
        kind = keyword ? "keyword_vararg" : "vararg"
    end
    type = "Any"
    name = _binding_name(arg)
    if arg isa Expr && arg.head === :(::)
        if length(arg.args) == 1
            name = "_"
            type = _syntax(arg.args[1])
        else
            name = _binding_name(arg.args[1])
            type = _syntax(arg.args[2])
        end
    end
    name === nothing && (name = "_")
    return (name=name, type=type, kind=kind,
            default=default === nothing ? "" : _syntax(default))
end

function _unwrap_signature(signature)
    whereparts = String[]
    while signature isa Expr && signature.head === :where
        append!(whereparts, _syntax.(signature.args[2:end]))
        signature = signature.args[1]
    end
    if signature isa Expr && signature.head === :(::)
        signature = signature.args[1]
    end
    return signature, sort!(whereparts)
end

function _normalise_component_signature(value::AbstractString)
    parsed = try
        Meta.parse(value)
    catch
        return replace(strip(value), r"\s+" => " ")
    end
    return _syntax(parsed)
end

_normalise_component_signature(value) = _normalise_component_signature(_syntax(value))

function _record_tuple_signature!(out::Dict{String,Vector{String}}, value)
    value isa Expr && value.head === :tuple || return out
    signature = _normalise_component_signature.(value.args)
    out[join(signature, '\0')] = signature
    return out
end

"Record tuples returned explicitly anywhere in the method, excluding nested callables."
function _explicit_tuple_signatures!(out, ex; top::Bool=true)
    ex isa Expr || return out
    if ex.head === :return
        _implicit_tuple_signatures!(out, only(ex.args))
        return out
    elseif !top && ex.head in (:function, :->, :macro)
        return out
    end
    for arg in ex.args
        _explicit_tuple_signatures!(out, arg; top=false)
    end
    return out
end

"Record tuple alternatives produced by the result position of blocks/branches."
function _implicit_tuple_signatures!(out, ex)
    ex isa Expr || return out
    if ex.head === :tuple
        return _record_tuple_signature!(out, ex)
    elseif ex.head === :return
        return _implicit_tuple_signatures!(out, only(ex.args))
    elseif ex.head === :block
        values = filter(arg -> !(arg isa LineNumberNode), ex.args)
        isempty(values) || _implicit_tuple_signatures!(out, values[end])
    elseif ex.head === :if
        length(ex.args) >= 2 && _implicit_tuple_signatures!(out, ex.args[2])
        length(ex.args) >= 3 && _implicit_tuple_signatures!(out, ex.args[3])
    elseif ex.head in (:let, :try)
        # Each block in a result-producing let/try can be a return alternative.
        for arg in ex.args
            arg isa Expr && arg.head === :block &&
                _implicit_tuple_signatures!(out, arg)
        end
    end
    return out
end

function _documented_tuple_signatures(doc::Union{Nothing,String})
    doc === nothing && return Vector{String}[]
    result = Vector{String}[]
    for match in eachmatch(r"->\s*\(([^()]+)\)", doc)
        push!(result, _normalise_component_signature.(
            split(match.captures[1], ','; keepempty=false),
        ))
    end
    return result
end

function _tuple_component_signatures(body, doc)
    signatures = Dict{String,Vector{String}}()
    _explicit_tuple_signatures!(signatures, body)
    _implicit_tuple_signatures!(signatures, body)
    for signature in _documented_tuple_signatures(doc)
        signatures[join(signature, '\0')] = signature
    end
    return sort!(collect(values(signatures)); by=signature -> join(signature, '\0'))
end

function _method_record(signature, body, path, exports, doc)
    signature, whereparts = _unwrap_signature(signature)
    signature isa Expr && signature.head === :call || return nothing
    name = _binding_name(signature.args[1])
    name in exports || return nothing
    positional = NamedTuple[]
    keywords = NamedTuple[]
    for arg in signature.args[2:end]
        if arg isa Expr && arg.head === :parameters
            append!(keywords, (_argument(kw; keyword=true) for kw in arg.args))
        else
            push!(positional, _argument(arg))
        end
    end
    tuple_component_signatures = _tuple_component_signatures(body, doc)
    tuples = Set(length(signature) for signature in tuple_component_signatures)
    return (
        name=name,
        path=path,
        positional=positional,
        keywords=keywords,
        where=whereparts,
        tuple_arities=sort!(collect(tuples)),
        tuple_component_signatures=tuple_component_signatures,
    )
end

function method_fingerprint(method)
    pos = join((arg.kind * ":" * arg.type *
                (isempty(arg.default) ? "" : "=" * arg.default)
                for arg in method.positional), ",")
    kw = join(sort([arg.name * ":" * arg.type *
                    (isempty(arg.default) ? "" : "=" * arg.default) *
                    (arg.kind == "keyword_vararg" ? "..." : "")
                    for arg in method.keywords]), ",")
    wherepart = join(method.where, ",")
    return join((method.name, pos, kw, wherepart), "|")
end

"Convert one declaration record into stable TOML-compatible scalar arrays."
function method_to_fixture(method)
    positional = [join((arg.name, arg.type, arg.kind, arg.default), "\u001f")
                  for arg in method.positional]
    keywords = [join((arg.name, arg.type, arg.kind, arg.default), "\u001f")
                for arg in method.keywords]
    return Dict(
        "fingerprint" => method_fingerprint(method),
        "name" => method.name,
        "positional" => positional,
        "keyword" => keywords,
        "where" => method.where,
        "tuple_arities" => method.tuple_arities,
        "tuple_component_signatures" => method.tuple_component_signatures,
    )
end

function _fixture_argument(value)
    fields = split(value, '\u001f'; keepempty=true)
    length(fields) == 4 || throw(ArgumentError("invalid method argument fixture: $value"))
    return (name=fields[1], type=fields[2], kind=fields[3], default=fields[4])
end

"Reconstruct the comparable part of a declaration record from its fixture."
function method_from_fixture(entry)
    record = (
        name=entry["name"],
        path="fixture",
        positional=_fixture_argument.(entry["positional"]),
        keywords=_fixture_argument.(entry["keyword"]),
        where=String.(entry["where"]),
        tuple_arities=Int.(entry["tuple_arities"]),
        tuple_component_signatures=[
            _normalise_component_signature.(String.(signature))
            for signature in entry["tuple_component_signatures"]
        ],
    )
    method_fingerprint(record) == entry["fingerprint"] ||
        throw(ArgumentError("method fixture fingerprint drift for $(record.name)"))
    return record
end

function _argument_compatible(baseline, current, type_compatible)
    baseline.kind == current.kind || return false
    baseline.default == current.default || return false
    return type_compatible(baseline.type, current.type)
end

"""
    method_compatible(baseline, current; type_compatible=(a, b) -> a == b)

Return whether `current` accepts the declaration-level call contract recorded
by `baseline`. Extra current keywords are compatible; removing, narrowing, or
changing the default of a baseline keyword is not.
"""
function method_compatible(baseline, current;
                           type_compatible=(baseline, current) -> baseline == current)
    baseline.name == current.name || return false
    baseline.where == current.where || return false
    length(baseline.positional) == length(current.positional) || return false
    all(zip(baseline.positional, current.positional)) do (b, c)
        _argument_compatible(b, c, type_compatible)
    end || return false

    current_keywords = Dict(arg.name => arg for arg in current.keywords)
    for baseline_keyword in baseline.keywords
        current_keyword = get(current_keywords, baseline_keyword.name, nothing)
        current_keyword === nothing && return false
        _argument_compatible(baseline_keyword, current_keyword, type_compatible) ||
            return false
    end
    all(arity -> arity in current.tuple_arities, baseline.tuple_arities) ||
        return false
    current_tuple_signatures = Set(
        Tuple(_normalise_component_signature.(signature))
        for signature in current.tuple_component_signatures
    )
    all(baseline.tuple_component_signatures) do signature
        Tuple(_normalise_component_signature.(signature)) in current_tuple_signatures
    end || return false
    return true
end

function _replace_type_variables(ex, variables)
    ex isa Symbol && return get(variables, ex, ex)
    ex isa Expr || return ex
    return Expr(ex.head, (_replace_type_variables(arg, variables) for arg in ex.args)...)
end

"""
    resolve_type_expression(text, where_parameters) -> Union{Type,Nothing}

Resolve the deliberately small set of source-level argument types used by the
baseline inventory. Free `where` variables are replaced by conservative sample
types (`Float64`) or dimensions (`2`) solely for `hasmethod`/subtyping probes.
Unknown syntax returns `nothing` and must be covered by an explicit family
runtime probe instead of disappearing from the audit.
"""
function resolve_type_expression(text::AbstractString, where_parameters=String[])
    variables = Dict{Symbol,Any}()
    for parameter in where_parameters
        parsed = try
            Meta.parse(parameter)
        catch
            return nothing
        end
        name = parsed isa Symbol ? parsed :
               parsed isa Expr && parsed.head === :(<:) ? parsed.args[1] : nothing
        name isa Symbol || return nothing
        variables[name] = name === :N ? 2 : :Float64
    end
    parsed = try
        Meta.parse(text)
    catch
        return nothing
    end
    substituted = _replace_type_variables(parsed, variables)
    value = try
        Core.eval(SHTnsKit, substituted)
    catch
        return nothing
    end
    return value isa Type ? value : nothing
end

function _walk!(methods, macros, types, aliases, ex, path, exports;
                doc::Union{Nothing,String}=nothing)
    ex isa Expr || return
    if ex.head === :macrocall
        macro_name = _binding_name(ex.args[1])
        if macro_name == "@doc" || macro_name == "doc"
            payload = ex.args[end]
            text = length(ex.args) >= 3 && ex.args[end - 1] isa String ? ex.args[end - 1] : doc
            _walk!(methods, macros, types, aliases, payload, path, exports; doc=text)
            return
        end
        for arg in ex.args[3:end]
            _walk!(methods, macros, types, aliases, arg, path, exports; doc=doc)
        end
        return
    elseif ex.head === :function
        body = length(ex.args) >= 2 ? ex.args[2] : Expr(:block)
        record = _method_record(ex.args[1], body, path, exports, doc)
        record === nothing || push!(methods, record)
        return
    elseif ex.head === :(=)
        lhs, _ = _unwrap_signature(ex.args[1])
        if lhs isa Expr && lhs.head === :call
            record = _method_record(ex.args[1], ex.args[2], path, exports, doc)
            record === nothing || push!(methods, record)
            return
        end
    elseif ex.head === :macro
        name = _binding_name(ex.args[1])
        exported = name === nothing ? nothing : "@" * name
        exported in exports && push!(macros, exported)
        return
    elseif ex.head in (:struct, :abstract, :primitive)
        name = _binding_name(ex.args[ex.head === :struct ? 2 : 1])
        name in exports && push!(types, name)
    elseif ex.head === :const
        assignment = only(ex.args)
        if assignment isa Expr && assignment.head === :(=)
            name = _binding_name(assignment.args[1])
            name in exports && push!(aliases, name)
        end
    end
    for arg in ex.args
        _walk!(methods, macros, types, aliases, arg, path, exports; doc=doc)
    end
end

"""
Build a deterministic declaration inventory. Entries not represented by a direct
method, macro, type, or constant alias are retained as explicit runtime probes;
this includes extension fallbacks produced by generated or interpolated syntax.
"""
function inventory_sources(sources, exported_names)
    exports = Set(String.(exported_names))
    methods = NamedTuple[]
    macros = Set{String}()
    types = Set{String}()
    aliases = Set{String}()
    parse_errors = NamedTuple[]
    for (path, source) in sort!(collect(sources); by=first)
        parsed = try
            Meta.parseall(source; filename=path)
        catch err
            push!(parse_errors, (path=path, error=sprint(showerror, err)))
            continue
        end
        _walk!(methods, macros, types, aliases, parsed, path, exports)
    end
    unique_methods = Dict(method_fingerprint(method) => method for method in methods)
    methods = sort!(collect(values(unique_methods)); by=method_fingerprint)
    direct = Set(method.name for method in methods)
    probes = [(name=name,
               kind=name in macros ? "macro" :
                    name in types ? "type" :
                    name in aliases ? "alias" :
                    name in direct ? "method_family" : "generated_or_extension")
              for name in sort!(collect(exports))]
    source_digest = bytes2hex(sha256(join(
        (path * "\0" * source for (path, source) in sort!(collect(sources); by=first)),
        "\0",
    )))
    return (methods=methods, probes=probes, parse_errors=parse_errors,
            source_digest=source_digest)
end

end
