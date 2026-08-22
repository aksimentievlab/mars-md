# Constants.h
## langevin_damping_unit = 2.3900574e-9f;
  Converts transDamping/rotDamping from config units (1/ns, acting on a omentum in amu AA/ns) into kcal/mol/AA. Legacy applies this ONCE at setup (RigidBodyType::setDampingCoeffs), so every later use of transDamping/rotDamping there is already scaled. arbd2 stores them unscaled, so it must be applied at use - in BOTH the noise coefficient and the drag term, exactly as legacy's addLangevin sees them. Omitting it entirely is what let the drag dwarf the momentum it acted on and diverged every rigid body to NaN.

# SimManager.cpp

## build_structure_view() — how a rigid body is identified in the output PSF/PDB

Cosmetic atoms used to get `segname = "RB" + rb.id`, which named the instance and
said nothing about the type. That is backwards for the thing the segname is
actually used for: a VMD representation. A QuickSurf rep already separates
instances on its own — they are spatially disjoint, so one rep over all of a
type's atoms draws every body — whereas telling `1ema` from `4hhb` in a
71-body, 6-type system has no other handle.

So the two facts are split across two fields:

- `segname` = `rigid_body_segname(type.name)`, the type name up to its first
  `.`, capped at four characters. `1u8f0.protein` -> `1u8f`. Four is the width
  of the PDB segID column (73-76); anything longer cannot survive a PDB round
  trip, and the PSF's own field is only seven.
- `beta` = `rb.id`, so an individual body is still selectable (`beta 7`) and
  `Color by Beta` shades a field of bodies by instance.

Attached particles keep `segname ATT` — that marks them as real, force-bearing
particles, which is the more important distinction — and also carry their owner
in `beta`, so `segname ATT and beta 7` is body 7's attached set.

`refresh_structure_positions()` only rewrites positions, so beta survives every
frame.

# IO/PsfPdbIO.cpp

## write_pdb_atoms() — the segID column was in the wrong place

The trailing format was `"...%6.2f%6.2f  %2s%6s\n"`: two literal spaces, an
always-empty `%2s`, then segname right-aligned in six. That puts segname in
columns 71-76 (1-based), while `parse_pdb_atom_line()` reads it from 73-76
(`substr_field(line, 72, 4)`). A segname of four characters or fewer happened to
land correctly because right-aligning it in six pushed it to 73-76; five or six
characters shifted left and read back mangled.

Now `"...%6.2f%6.2f      %-4.4s\n"` — six spaces, then segname left-aligned in
four at 73-76, which is where the standard puts segID and where the reader looks.
Same total record length (76), and the truncation cap moved from 6 to 4 to match.

## init() — the bonded-interaction guard must name every term

`src/arbd.cpp` drives the executable through `set_bonded_interactions()`, not
`load_config()`, so the staged `pending_bonded_interactions_` only reaches
`sys_state_` if `init()`'s guard says it is non-empty. That guard tested bonds,
angles, dihedrals and exclusions — not restraints. A config whose *only* bonded
term is restraints (`Tests/privite_test/npc_6enl_beads`, 437 of them and nothing
else) therefore lost all of them between the parser and the patch, while
anything with a bond count sailed through.

Any new term added to `BondedInteractions` has to be added to that guard too.
The `!...empty()` disjunction is the whole gate; there is no other path.

## SimManager::run — DLM half-kicks straddle the force phase (2026-08-16)

`run()` orders the loop as drift -> forces -> kick, not forces -> integrate.

Legacy is laid out the same way: `GrandBrownTown.cu:837-838` runs
`integrateDLM(0)` and `(1)`, then `:953` clears, `:1097` re-evaluates forces,
`:1130-1131` redraws the Langevin noise, and `:1132` runs `integrateDLM(2)`.
Forces are primed before the loop at `:694` / `:775` / `:789-790`.

Running substeps 0-2 back to back leaves both half-kicks on a pre-drift force.
For a harmonic mode that map is `[[1 - w^2 dt^2/2, dt], [-w^2 dt, 1]]`, whose
determinant is `1 + w^2 dt^2 / 2 > 1`, so it pumps phase-space volume at a rate
set by that mode's own stiffness — no conserved energy, and anisotropic heating
when the stiffness is anisotropic.

This is **not** a second force evaluation per step. The evaluation moves between
substeps 1 and 2. The only added cost is the one priming call before the loop.

Brownian is excluded (`split_dlm`): the overdamped path takes one force
evaluation per step by construction and has no half-kicks.

**Unverified assumption:** the priming call re-runs the whole force phase, so it
assumes a force evaluation is idempotent. `Patch::calculate_bonded_forces` opens
with `particles_.clear_forces()` and RB forces are cleared at the top of
`execute_force_calculation`, but nothing has confirmed the full pipeline yields
identical buffers when called twice with no integration between. Check before
trusting step 1.

## execute_force_calculation — buffer clear order and energy gating

`calculate_nonbonded_forces()` clears the force buffer and writes the PMF/grid
plus pairwise nonbonded terms. `calculate_bonded_forces()` must run after it and
accumulate on top; it only ever atomic-adds and never clears.

Energy accumulation costs an extra atomic write (`ForceEnergy.t`) alongside the
three force-component atomics on every pairwise-neighbour and bonded-term
evaluation. It is therefore enabled only on steps `handle_output()` will report
energy for, matching legacy ARBD, which likewise computes energy every
`outputEnergyPeriod` steps rather than every step.

## handle_output — paying BAOAB's deferred kick before reading momentum

BAOAB defers its closing half-kick to the top of the next step (see
`Patch::finish_deferred_kick`), so at output time every particle's momentum is
half a kick behind its position. Positions are unaffected, but momentum DCD
frames, restart files and kinetic energies read that momentum and would record —
and on restart resume from — a half-step value.

So the kick is paid before reading. It needs the force at the current positions,
which nothing has evaluated yet, hence the extra force calculation first. The
next step would have recomputed exactly that anyway, so the cost is one extra
force evaluation per output frame (~0.1% at `outputPeriod 1000`).
`finish_deferred_kick()` clears the pending flag, so the next step applies only
its opening half-kick and the total impulse is unchanged.

## SimManager implementation notes

### Initialization

Initial particles are staged until particle type IDs and name maps exist. Bonded
interactions must be copied when any term is present, including restraints.
Rigid-body grids and dispatch tables are prepared only after device grid arrays
are built. Rigid-body external loads cross the IO boundary from AoS vectors to
device-side SoA vectors. Attached-particle ranges come from the parsed
`RigidBodyIO` instances, while the device manager consumes the converted state.
Structure files are written during initialization so interrupted runs still have
matching PSF and PDB files; visualization-write failures remain non-fatal.

### Force and integration ordering

The force phase clears rigid-body forces, refreshes attached positions, computes
nonbonded terms before bonded terms, then computes rigid-body grid and particle
forces before Langevin forces. Attached-particle forces are reduced onto their
parent bodies. Particle-type device tables are cached per resource because
particle types remain static. Force events must finish before integration reads
them or the next force phase overwrites them. Linear interpolation remains the
configured default; energy accumulation is enabled only on energy-output steps.

Integration waits for each patch event before the next force phase. DLM runs
around the force phase in `run()`; Brownian integration runs in this method.

### Output and structure layout

Momentum output is needed only for Langevin dynamics. A DCD frame contains
regular particles, attached particles, and cosmetic rigid-body atoms in that
order. Periodic DCD frames include the orthorhombic CHARMM unit-cell block.
The structure view maps real bonded topology directly and remaps each cosmetic
template's bonds into the appended atom range. Attached atoms use `ATT` and
cosmetic atoms use the rigid-body type segment name; rigid-body instance IDs are
stored in beta. An empty structure is an error rather than a cached no-op.

Rigid-body trajectory keys use `<typeName>#<index within type>`, with instances
numbered independently for each type. Its orientation is emitted row-major
from the column-oriented `Matrix3`; the final six legacy-labelled fields are
raw linear and angular momenta. Momentum DCD output reuses the particle gather
performed for the position frame.

### Deferred momentum and energy

BAOAB's closing half-kick is paid before momentum DCD, restart, or kinetic-energy
output. It requires a force evaluation at the current positions. Particle
kinetic energy is divided by `SQRT_CAL_TO_JOULE^2` before conversion to kT.
Rigid-body energy output currently writes zero values for compatibility with
the legacy file format.

### Restart and synchronization TODOs

Restart formatting and file writes run asynchronously, with one pending write at
a time. Langevin momentum restart keeps the legacy `.0.momentum.restart` name.
Replica count is currently fixed at one and should be exposed by `SimSystem`.
Multi-resource halo exchange remains unimplemented and should use
`PatchManager::exchange_halos()` when available. IMD disconnect, command
polling, and state updates remain unimplemented.

### Initial momentum and reactions

Initial momentum generation samples Maxwell-Boltzmann components, removes the
center-of-mass momentum, and adds the requested center-of-mass velocity. Grid
temperature is unsupported, and the generated vector is not yet routed into
initial-state storage. Particle reactions remain a design stub: run the reaction
kernel, compact dead particles, remap topology, append births, and rebuild the
neighbor list.
