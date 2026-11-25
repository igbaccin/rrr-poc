# Profiles

RRR ships three profiles from one repository.

| Profile | Runtime | Source |
| --- | --- | --- |
| `vlm` | Local models through Ollama | `src/rrr` |
| `vcx` | Codex subscription (plugin) | `src/rrr` with `profiles/hosted/rrr` laid on top |
| `vcl` | Claude subscription (skill) | `src/rrr` with `profiles/hosted/rrr` laid on top |

The local writer carries extra instructions and repair passes for the smaller
models. The hosted writer is leaner. Every other module is shared, so a fix to
ingest, preprocessing, the command line or the host adapter is written once in
`src/rrr` and reaches all three profiles.

`profiles/hosted/rrr` holds only the modules that differ in the hosted
profiles:

- `writer.py`, the hosted writer;
- `render.py`, which differs from the local module in how it reads a grouped
  narrative citation; and
- `reasoner.py`, which differs in a comment.

These three files entered the repository as exact copies of the modules inside
`rrr_poc-15.19.3-py3-none-any.whl` (vcx.2.3 and vcl.2.3).

A change to citation rendering or validation usually has to be made in both
`src/rrr/render.py` and `profiles/hosted/rrr/render.py`, and in both writers.
`tests/unit/test_hosted_profile.py` assembles the hosted tree and runs
`tests/hosted/` against it, so both sides stay under test.

Build the wheels with `scripts/build_product_distribution.py`. Its docstring
gives the commands.
