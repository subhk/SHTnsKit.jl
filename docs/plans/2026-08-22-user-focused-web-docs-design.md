# User-Focused Web Documentation Design

**Audience:** Scientists and Julia users reaching SHTnsKit for the first time.

**Goal:** Make the published site answer the common path—install, run a correct
transform, choose a grid, and find the relevant accelerator guide—without
presenting release evidence or implementation details as required reading.

## Information architecture

Use progressive disclosure. Keep **Installation** and **Quick Start** in a
small Getting Started group. Put the essential **Grid Types** guide (including
both responsive grid-pattern SVGs), a curated **Examples Gallery**, and the
GPU, distributed, performance, and advanced guides under User Guide. Keep
normalization, the generated API, and v1-to-v2 migration under Reference.

Do not add generated script pages, the duplicate performance-tips page, or the
SHTns 3.7 release-evidence page to public navigation. Their source files can
remain in the repository for maintainers and direct links.

## Editorial scope

- Home: one-sentence purpose, four practical capabilities, one installation
  command, one verified roundtrip, and a task-based route table.
- Quick Start: explain the two array layouts and one scalar roundtrip; send
  vector, batch, device, and plan users to focused guides instead of teaching
  every API on the first page.
- Installation: retain core and optional package installation plus short
  verification/troubleshooting; remove development-checkout instructions and
  low-level dispatch edge cases.
- GPU: retain vendor installation, device-resident transforms, reusable plans,
  and concise troubleshooting; omit compatibility-wrapper and utility catalogs.
- Distributed: retain when MPI helps, package setup, one accurate runnable
  roundtrip, the replicated-versus-distributed spectrum choice, and launcher
  guidance; remove duplicated examples, marketing claims, internal limitations,
  and API tables already generated elsewhere.
- Examples: keep a small set covering scalar transforms, spectra, vector
  decomposition, stream functions, benchmarking, and rotation. Link to the two
  maintained MPI scripts instead of embedding three long MPI programs.

## Acceptance criteria

The grid figure remains visible at desktop and mobile widths. Every published
example executes successfully, internal links resolve during a strict
Documenter build, the main navigation contains no duplicate or maintainer-only
pages, and the package regression suite continues to validate gallery code.
