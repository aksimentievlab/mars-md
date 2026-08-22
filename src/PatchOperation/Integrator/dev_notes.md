# RBBD.h — implementation notes

`RBIntegrateBDKernel` — overdamped Brownian rigid-body dynamics, ported from
legacy `RigidBody::integrate` (`arbd_chris/src/RigidBody.cu`). One kernel, one
full step, no momentum.

## Why it exists

Legacy has two RB integrators and Brownian is the **default**:

- `Configuration.cpp:946` — `RigidBodyDynamicType = String("Brown")`
- `GrandBrownTown.cu:924` — `"Langevin"` → `SetRandomTorques()` + `AddLangevin()`,
  then `integrateDLM(sys,0)` / `integrateDLM(sys,1)`
- `GrandBrownTown.cu:936` — **else** (i.e. `Brown`) → `RigidBody::integrate(sys, s)`

arbd2 had only the DLM path, so every `rigidBodyDynamicType` other than
`Langevin` logged a warning and silently ran DLM instead.

## Where the external load is applied

Legacy adds `constantForce`/`constantTorque` in
`RigidBodyController::updateForces` (`RigidBodyController.cu:586-587`), inside
the grid–particle force loop that runs for **both** integrators — not inside
`addLangevin`.

arbd2 had folded it into `RBLangevinForceKernel` instead, which would have
dropped it entirely on this path: `RigidBody::integrate` consumes `force`
directly and never calls `addLangevin`. So the `+= external_force/_torque` moved
out of the thermostat kernel and into each integrator:

- `RBIntegrateDLMKernel`, substeps 0 and 2 (the half-kicks — the only places
  that read `force`)
- `RBIntegrateBDKernel`, once per step

Applied exactly once either way, and no extra kernel launch. Behaviour is
unchanged for existing DLM runs, since `add_langevin_forces()` always ran there.
`DeviceRigidBody::clear_forces()` never touches `external_*`, so a published
load persists until its owner overwrites it.

## Units

Legacy scales the damping coefficients once at setup
(`RigidBodyType::setDampingCoeffs`: `transDamping = 2.3900574e-9 * transDamping`),
so every later use in legacy is already scaled. arbd2 stores them unscaled, so
`constants::langevin_damping_unit` is applied at use.

`constants::langevin_damp_scale` (the literal `10000`) is **not** applied here —
it belongs to `addLangevin`'s drag term, which this path never runs.

`kT` is legacy's `Temp` (`RigidBody.cu:28`: `temperature * 0.0019872065`), i.e.
kcal/mol, not Kelvin.

Legacy's `diffusion / Temp` is formed directly as a mobility
(`1/(damping*mass)`), partly to skip a redundant multiply and divide by `kT`, and
partly because `Vector3.h:311`'s `operator/(scalar, Vector3)` declares its return
type as `T` while returning a `Vector3_t<TU>` — it does not compile if
instantiated.

## Zero damping — checked at setup, not in the kernel

Mobility is `1/(damping*mass)`, undefined at zero. `trans_damping`/`rot_damping`
**default to zero** (`RigidBodyProperties.h:60-61`) while `mass` and `inertia`
carry explicit non-zero defaults, so a type that simply omits `transDamping` — a
legal config — gives an infinite mobility and a whole-trajectory NaN on step 1.

`RigidBodyType::check_damping(IntegratorType)` rejects it at setup, called from
`SimManager` over every type once the manager is built. Setup is the right place:
these are per-type constants, so a kernel-side branch would cost a test per body
per step to re-derive something already known, and would have to *invent* a
fallback — freeze the body, or let it fly — where no correct answer exists.
Failing at setup names the type and the offending numbers instead.

It also checks `mass`/`inertia` for **both** integrators, since DLM divides by
them too (`RBDLM.h:132,135`).

## RNG stream separation

Philox is counter-based: `Philox(seed, ctr, global_seed, ctr1)`. Two kernels
sharing all four arguments produce identical numbers, so stream identity has to
live in an argument — not in an index offset.

Adding a constant to the index (`idx + 23`) does **not** work. Particle
`BDIntegrate` uses `(base_seed + step, base_ctr + idx)`; an RB kernel using
`(base_seed + step, idx + 23)` still collides, just shifted — body 0 now draws
what particle 23 draws, body 1 what particle 24 draws, and so on. Since particle
counts far exceed body counts, essentially every body still overlaps some
particle. Any fixed offset smaller than the other stream's index range only moves
the collision.

This kernel passes a distinct `ctr1` instead (`rng_stream = 0x52424244`, vs
openrand's `0x12345` default), which separates the stream at every index at once.

`RBLangevinForceKernel` still uses the default `ctr1` and so still overlaps the
particle integrators. Left alone deliberately: changing it changes every existing
DLM trajectory, and the v1 thermostat comparison (~3%) is still open. Worth
tagging once that lands.

## Known gaps

- **No NaN guard.** Legacy throws `NaNError` and exits on a NaN force/torque
  (`RigidBody.cu:474`). Not reproduced: a device kernel cannot throw, the guard
  is diagnostic rather than corrective, and the realistic NaN source (zero
  damping) is now caught at setup. If step-level detection is ever wanted, the
  idiom already in the tree is a device flag polled every N steps, like
  `RigidBodyManager`'s `grid_grid_overflow_`.
- **Untested.** Not compiled or run.

## Philox argument mapping — seed and step must not share a word (2026-08-16)

`Philox(uint64_t seed, uint32_t ctr, uint32_t global_seed, uint32_t ctr1)`
(`extern/random/OpenRAND/include/openrand/philox.h:64`) splits `seed` into
`seed_hi`/`seed_lo` — the **key** — and fills the counter from
`ctr0 = ctr`, `ctr1 = ctr1`, `ctr2 = global_seed`.

All four integrators used to pass `base_seed + current_step` as the key. That
makes the seed and the step interchangeable:

    run A, seed S,   step s+k   ->  key S + s + k
    run B, seed S+k, step s     ->  key S + s + k

so run B is run A's noise sequence shifted `k` steps. An ensemble over seeds
`S..S+29` is therefore **one** realization offset by at most 29 steps, not 30
independent ones. Measured on the `npc_6enl_beads` torque-free campaign: 30 v2
runs tracked each other frame by frame for 200,000 steps, and the between-seed
scatter of `<L_x^2>` was 0.67% against the ~7% a real ensemble must show. Every
error bar computed from seed-to-seed scatter before this date is meaningless.

`BAOAB` had the sharper version of the same mistake: key `base_seed` alone with
counter `base_ctr + idx`, and `base_ctr` is a literal `0` at every call site
(`Patch.cpp:291,318,354,372,410`), so it had **no step dependence at all** and
redrew identical numbers every step.

Now each of the four inputs owns a word:

| word | holds |
| --- | --- |
| key (`seed`) | `base_seed` — the run |
| `ctr0` (`ctr`) | particle/body index (`base_ctr + idx` where present) |
| `ctr2` (`global_seed`) | `current_step` |
| `ctr1` | `rng_stream` — the per-kernel constant below |

`openrand::DEFAULT_GLOBAL_SEED` is no longer passed; the run seed already
separates applications, and that slot is worth more as the step. The stream
constants stay where "RNG stream separation" above put them, in `ctr1`.

This changes every stochastic trajectory, so every saved reference needs a
re-baseline.
