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

## ConfigParser.cpp — inputParticles + restartCoordinates loaded the system twice

A `.bd` with both `inputParticles` and `restartCoordinates` produced double the
particles. `load_particles_file()` and `load_restart_file()` both `push_back` into
the same `init_particles_`, so 65,248 beads from the particle file plus 65,248 from
the restart = 130,496 — every bead duplicated on top of its original. The overlap
made the nonbonded force diverge and the system exploded (huge KE, structure flew
apart). `run-hinged-nup98/nupod-0-cont01.bd` was the case: the true count is 65,248
(the `num` sum and the restart line count), and MARS loaded 130,496 — exactly 2x.

ARBD never adds a second copy: `inputParticles` defines the set; `restartCoordinates`
(`loadRestart`) overwrites those particles' positions in line order. `has_explicit_
particle_source` already suppresses the per-type placeholder path when either key is
present, so the duplication was purely the two file loaders both appending.

Fix, matching ARBD:

- **`load_restart_file` overwrites when `out` is non-empty** — line index N updates
  the staged particle N's position, keeping its `inputParticles` type_name and
  momentum. Empty `out` (restart is the sole source) keeps the old create-and-append
  path. Count mismatch between restart lines and staged particles logs a WARN.
- **`restartCoordinates` is deferred** to after the first parse pass (like
  `inputRBCoordinates` → `rb_coordinates_file`), stored in `restart_coordinates_file`
  and applied at the end. This makes the overwrite independent of whether
  `restartCoordinates` appears before or after `inputParticles` in the file.

Line-order pairing is safe because both files are written in particle-id order
(`particles.txt` has an explicit id column 0..N-1; the restart is line-ordered).

Companion change: `SimManager::generate_initial_momentum` now actually runs for
Langevin runs that were given no momentum (Maxwell-Boltzmann seed, ARBD
`Configuration::Boltzmann`). It was dead code before — never called, discarded its
result, and indexed the type list by particle index. Unit test:
`Unit_test/System/BoltzmannMomentumInit.cpp`.

## Bidirectional bond dedup (`BondConfigReader::parse_bond_line`)

Legacy ARBD bond files list most bonds twice — both `(i j)` and `(j i)` with the
same spring file. `TabulatedBondComputer` applies Newton's 3rd law per entry, so
processing both entries adds the identical force/energy to each endpoint twice:
the bonded term ends up ~2x. ARBD hides this by launching its bond kernel over
`numBonds/2`; MARS reads every `BOND` line, so it must dedup itself.

Fix: keep one bond per unordered pair. `seen_bond_slot_` maps `canonical_bond_key`
(the `(min,max)` index pair packed into a `uint64_t`) to the bond's slot in
`bonds_`. Identical duplicates are dropped silently; a duplicate with a *different*
potential warns and overwrites (latest wins) via `BondedInteractions::set_bond`.
The pair's exclusion is added only on first insert, so REPLACE exclusions aren't
duplicated either.

Symptom this fixed: hinged-nup98 Langevin runs heated to ~4x equipartition KE and
blew up, while Brownian (unconditionally stable) only descended a stiffer surface.
Diagnosis was the ~1.85x bonded force from `nupod-0.bond.txt` (853,964 lines /
460,316 unique pairs; 85% doubled, 0 conflicting springs); measured PE ratio
v2/v1 ≈ 1.82 matched the 1.855 duplication factor. Heating scaled as 1/gamma
(velocity-Verlet injection ∝ dt²k/m dissipated ∝ gamma): transDamping-20 protein
surfaced it, transDamping-400 beads stayed at 0.5 kT/dof. Angles (`nupod-0.angle.txt`)
are not doubled. Dihedrals show a ~1.03x duplication with some reversed
`(l k j i)` entries carrying a *different* table — left alone, since a reversed
dihedral samples -phi and may be a legitimate second term, not a duplicate.

### Duplicate-bond summary warning (2026-09-16)

`read_file` now ends with a single `LOGWARN` whenever any BOND line was collapsed:
count, total, percentage, unique kept, and how many carried a conflicting
potential. Counters (`bond_lines_`, `duplicate_bonds_`, `conflicting_bonds_`)
reset per `read_file` call; `seen_bond_slot_` deliberately does not, so dedup
still spans multiple bond files. Without this the dedup was silent — a malformed
generator could double every bond and nothing in the log would say so.

### Corrections to the section above, from a direct v1/v2 audit

Two claims above are wrong and should not be reused:

1. **"measured PE ratio v2/v1 ~= 1.82 matched the 1.855 duplication factor"** — no.
   Converged bond-only runs on hinged-nup98 give v1 271k vs v2 247k, a ratio of
   **1.09**, not 1.82. Bond energy is a near-blind detector of double-counting:
   - The doubled pairs are the stiff structural bonds, sitting at the bottom of
     their wells: 393,648 of them hold 25.7k (0.065 each), while the 66,668
     singly-listed bonds hold 257.9k (3.87 each). Doubling ~0 gives ~0.
   - Equipartition then *cancels* what is left. <U> = kT/2 per bond DOF regardless
     of force constant, so doubling the stiffness halves the mean energy per bond:
     measured 0.0653 (v1) vs 0.1635 (v2) per doubled bond, ratio 0.40 vs the 0.50
     predicted. v1's doubled contribution (2 x 25.7k = 51.4k) is *smaller* than
     v2's single one (64.4k). The direct energy signature is negative.
   - The residual +22k gap is indirect: v1's over-stiff network strains the floppy
     singly-listed bonds, which hold ~90% of the energy (257.9k vs 222.9k).

   The observable that actually resolves it is the per-bond thermal fluctuation
   amplitude with the singly-listed bonds as control: v1/v2 = **0.805** on doubled
   pairs vs **1.022** on the control (1/sqrt(2) = 0.707 if perfectly relaxed).
   Do not use the sd of bond length *across* bonds — that is ~2.05 A in both codes,
   dominated by the static spread of rest lengths between bond types, and has no
   sensitivity at all.

2. **"ARBD hides this by launching its bond kernel over `numBonds/2`"** — the
   `/2` and the `ind1 < ind2` filter in `GrandBrownTown.cu` undo `readBonds`'
   own deliberate double-append (one line -> `(i,j)` and `(j,i)`). They filter on
   *direction*, not identity. A pair written twice as reciprocals becomes 4 Bond
   entries and survives as 2. v1 has **no** dedup anywhere: no `std::unique`, no
   set, `numBonds` is never reduced, and `addExclusion` does not dedup either.
   Confirmed from v1's own instrumented `bondlist_dump.txt`: 66,668 pairs once,
   393,648 twice. So MARS's dedup is a deliberate divergence from v1, not a
   restoration of v1 behaviour.

The 4x Langevin heating was **not** caused by the duplication. Root cause was the
`int3` stride bug in `remap_particle_indices` (see
`PatchOperation/ZOrderKernels/dev_notes.md`).
