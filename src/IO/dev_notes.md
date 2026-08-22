# src/IO — dev notes

## RigidBodyVisualize.h — undeclared resnames on cosmetic template atoms

`RigidBodyPdbPsfReader::load()` used to resolve *every* template atom's resname
against the declared `ParticleType`s and throw when one did not match. That is
the right rule for an atom marked `segname ATT`, because such an atom becomes a
real `ParticleIO` whose type is looked up by resname. It is the wrong rule for
the rest of the template.

A cosmetic atom carries no physics: `prepare_cosmetic_atoms()` stores only its
body-frame offset, `build_structure_view()` writes it with `mass = 0` and
`charge = 0`, and it never enters a pair list or a bonded term. Its resname is
used for one thing — the resname column of the output PSF. So requiring a
declared type for it forced configs to declare particle types purely to satisfy
the reader.

`Tests/privite_test/npc_2022` is the case that surfaced this. Its particle table
was deliberately trimmed to `ALA`, `CONFINE`, `dummy` (every NB pair there is
`pot-zero`, and the per-residue potential grids for the other 19 types are 1.9 GB
that were never copied). The rigid body's template is the full 2hiu insulin
structure, 793 atoms over 17 resnames, of which exactly 2 are attached. Under the
old rule the fixture could not load its own rigid body without re-declaring 16
types it has no potentials for.

The check now runs only inside the `segname == ATT` branch, and its message says
"attached particle resname" so the failure names the actual constraint.

### The cosmetic type marker

An atom whose resname resolves keeps its PSF `type_name`. One that does not gets
`constants::kCosmeticTypeName` (`"COS"`) instead, so the two groups are
distinguishable downstream — in practice, selectable as a group in VMD:

```
segname RB0 and type COS
```

Keeping the atom's own `name`, `resname` and `resid` means the structure still
reads as a protein; only the type column is overwritten, and only for atoms that
had no type to begin with.

## PSF/PDB round trip for rigid-body templates

`Tests/privite_test/tools/make_v2_rb_templates.py` writes template PSFs in
exactly the format `write_psf_topology()` emits (`PSF NAMD`, free-form bonds,
fixed 8-column angle/dihedral/improper blocks), because that is the format
`read_psf_file()` is guaranteed to accept back. Two constraints worth recording:

- **Segname must never be empty.** The NAMD-format atom line is parsed with
  `sscanf("%d %15s %15s %15s %15s %15s %f %f")`. `write_psf_topology()` writes
  segname as `%-7.7s`, so an empty segname collapses to whitespace and every
  later field shifts left by one.
- **The PSF wins on resname and segname.** `merge_pdb_coordinates()` takes only
  position, occupancy and beta from the PDB, and fills chain/segname only when
  the PSF left them blank. So a resname longer than the PDB's 3-column field —
  `CONFINE` — survives in the PSF even though the PDB shows `CON`.

## BondConfigReader.h — the RESTRAINT branch was missing

`read_file()` dispatched on `ANGLE`, `DIHEDRAL`, `BOND` and `EXCLUDE`.
`parse_restraint_line()` was fully written, correct, and unreachable: no key ever
routed to it, so `inputRestraints` files were opened, tokenized by `Reader`
(which logs a cheerful "Successfully parsed N parameter lines"), and thrown away
a line at a time.

That log line is why this hid for so long — it reports what `Reader` tokenized,
not what any consumer accepted. When chasing a dropped input, check the count
that the *consumer* reports, not the reader's.

Field order is v1's, from `Configuration.cpp:1911`:

```
RESTRAINT | INDEX1 | k | x0 | y0 | z0
```
