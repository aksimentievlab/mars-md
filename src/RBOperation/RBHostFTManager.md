# RBHostFTManager.h — implementation notes

`RBHostFTManager` is the single path for host-computed per-rigid-body external
force/torque reaching the device. It stages an index list plus force/torque into
device buffers and launches `ApplyExternalForcesKernel`, which scatters by
rigid-body index — so only the listed bodies are written and the rest keep
whatever they held.

## Why push() takes plain arrays, not a ScuffRigidBody

SCUFF is one producer, not the interface. Taking `const ScuffRigidBody&` would
make every other host-set force construct a scuff struct to set a number, and
would make this header — and anything including it — unbuildable without
`USE_SCUFF` and a linked scuff-em.

So `push()` takes `(indices, force, torque, n, rb)` and this file has **no
scuff-em dependency**. `ScuffRigidBody` calls it with its own vectors' `.data()`.

## Why external_force rather than force

`DeviceRigidBody::clear_forces()` zeroes `force_`/`torque_` every step but
deliberately leaves `external_*` alone, so one push persists until its owner
overwrites it. Writing to `force` instead would make every host-set load a
one-step impulse. Legacy gets the same persistence by re-adding `constantForce`
inside `updateForces` every step (`RigidBodyController.cu:586-587`); this achieves
it with a single upload.

The integrators fold `external_*` into the force they read
(`RBIntegrateDLMKernel` substeps 0/2, `RBIntegrateBDKernel`), so a pushed load
reaches every integrator, not only the thermostatted one.

## Producers compose through a host baseline, because the kernel assigns

`ApplyExternalForcesKernel` does `rb.external_force[id] = ...` — an assignment, not
an accumulation. With one producer that is fine. With two it is a bug: SCUFF's push
would overwrite a body's `constantForce`, silently losing it on the first publish.

`+=` in the kernel is not the fix. `clear_forces()` deliberately never zeroes
`external_*` (that is what makes a published load persist), so `+=` would
accumulate without bound.

So the composition happens on the host, as `ApplyHostForce.h`'s own
`@todo Needs to accumulate ALL forces on HOST first` says:

- `set_baseline(bodies)` — the always-on contribution, i.e. `constantForce`/
  `constantTorque`. Static for the run, so it is held here rather than re-sent.
- `push_baseline(rb)` — send the baseline alone. Call once after `set_baseline()`
  so constants land even if no dynamic producer ever pushes.
- `push_with_baseline(idx, f, t, rb)` — what a dynamic producer (SCUFF) calls.
  Sends `baseline + contribution` for the listed bodies, so the constant survives.
- `push(idx, f, t, rb)` — raw, baseline-free. For a producer that genuinely owns
  the whole value.

| Producer | Path |
| --- | --- |
| `constantForce`/`constantTorque` | `RigidBodyManager::set_external_loads()` → `set_baseline()` + `push_baseline()`, once at setup |
| scuff-em plasmonic force/torque | `ScuffRigidBody` + `ScuffForceCalculator` → `push_with_baseline()` per publish |

A third producer would want the baseline generalized into a full accumulation
buffer (every producer writes into a full-length host mirror, one push sends the
dirty union). Not built: with two producers it is speculative, and the baseline
split already spans the static/dynamic distinction that actually exists.

## No IO types cross into here

`set_baseline()` and `push()` take `std::span<const Vector3>`, never
`std::vector<RigidBodyIO>`. `RigidBodyIO` is a parse-time structure; a device
transfer manager taking one would drag the IO layer into the device path and force
every other producer (Python, a test, a tweezers module) to build full IO structs
just to set a force.

The AoS -> SoA extraction happens once at the IO boundary, in `SimManager` where
the `RigidBodyIO` list already lives — the same place `SystemState` converts the
fields `HostRigidBodyData` does carry. `RBOperation/` references no IO type at all.

## Aliasing

`ExternalForceView` and `RigidBodyView` both mark `external_force`/`external_torque`
`__restrict__`, and the kernel reads one while writing the other. They are distinct
allocations — the staging buffers here versus `DeviceRigidBody`'s — so the promise
holds. Constructing an `ExternalForceView` from `DeviceRigidBody::external_force()`
would make it a lie and is undefined behaviour; the fields share a name and type, so
this is easier to do by accident than it looks.

Duplicate indices within one push are also unchecked: two threads plain-store to the
same slot and the winner is undefined. Producers must send each body at most once.

## Staging-buffer reuse

`push()` overwrites `id_`/`force_`/`torque_` and then launches. This is safe only
because `DeviceBuffer::copy_from_host` defaults to `sync = true`, so the H2D
completes before the next launch is enqueued. Switching those to
`copy_from_host_async` would let push N+1 overwrite the buffers while push N's
kernel is still reading them.

## Torque is not hypothetical

Host-set per-body **torque** has real uses beyond scuff-em, all in ARBD's domain:

- **Magnetic tweezers** — twisting DNA with a rotating field is a standard
  single-molecule assay; the bead takes a host-computed torque.
- **Rotary motors** — F1-ATPase and the bacterial flagellar motor are routinely
  coarse-grained as a rigid body driven by a constant or duty-cycled torque.
- **Dipole in an applied field** — `torque = m x B` or `p x E` is constant for a
  fixed field, which is exactly what `constantTorque` expresses.
- **Optical torque** — circularly polarized traps; the same quantity scuff-em
  returns for plasmonic rotors.
- **Orientational steered MD / feedback control** — a host-computed restraint
  torque toward a target orientation, and the Python-driven equivalent after the
  migration.

## Why push() takes spans, not smart pointers

The array parameters are **borrows, not ownership**. A smart pointer parameter
would say "this call takes ownership of the allocation", which is false here and
would force every caller into one allocation strategy — no stack arrays, no
`vector` data, no subranges. Core Guidelines F.7 and R.30 say the same: take smart
pointers as parameters only to express lifetime transfer.

`std::span<const T>` is the right shape for a non-owning contiguous range, and is
already the house idiom (`TrajectoryWriter::writeNewFile`, `Reader::getParameters`,
`BaseGrid::span`). One overload then serves vectors, arrays and subranges, and the
size comes with the data instead of as a separate argument that can disagree.

The remaining raw pointers are correct as they are: `id_.data()` and friends are
**device** pointers handed to a kernel, and `ExternalForceView` is a non-owning
device view. Neither is a host allocation, so neither is a smart-pointer candidate.
Device memory ownership already lives in `DeviceBuffer`.
