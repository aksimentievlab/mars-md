# Tables.h implementation notes

## Angle and dihedral abscissa units

Tabulated angle and dihedral files are written in **degrees**:

```
# angle-20.801-180.000.dat        # dihedral-2.668-0.000.dat
0.000000 102.650772               -180.000000 13.167265
0.100000 102.536747               -179.900000 13.152639
...                               ...
181.000000 0.003168                180.000000 13.167265
```

The kernels index them with `acos(...)` and `atan2(...)` results, which are in
**radians**. `Table::read_file` therefore rescales `X`, `start` and `step_size`
by `PI/180` for `TabulatedType::Angle` and `TabulatedType::Dihedral` only.

Legacy applies the same factor, but folds it into the step instead of the data:
`TabulatedAngle.cu` computes

```
angle_step_inv = 57.29578f * (size-1) / (angle[size-1] - angle[0]);
```

where `57.29578 == 180/PI`. Rescaling the abscissa here is equivalent and keeps
`X`/`start`/`step_size` mutually consistent, which matters because
`check_same_step_size()` derives `start` and `step_size` from `X`.

### What this fixed

Without the conversion the lookup variable was in radians while the table was
laid out in degrees, so an angle table spanning 1811 rows was only ever sampled
over its first ~31 rows: `theta` in `[0, PI]` divided by a step of `0.1`
(degrees, read literally) gives indices `[0, 31.4]`. Every angle in the system
therefore evaluated at roughly `theta = 0` regardless of its real geometry —
a nearly constant energy of ~102.65 kcal/mol per angle and an essentially
meaningless force.

On the rotor fixture (`~/server3/rotor/rotor_center_debye_30ms2`, 5682 angles)
that showed up as ~571,000 kcal/mol of angle energy, against a v1 total system
potential of about -1,400. Dihedrals had the same defect: the table starts at
-180 (degrees) while the kernel looks up `phi + PI` in radians, so the sampled
index sat near the row for 0 degrees.

The `Table(TabulatedType::Dihedral)` constructor seeds `start = -PI`, but
`check_same_step_size()` overwrites it with `X[0]` once the file is read, so
that seed never protected against this.
