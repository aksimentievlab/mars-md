"""Direct CPU pair sum reference for a tabulated-nonbonded MARS config.

Independent check on mars2's reported nonbonded (and tabulated-bond) energy:
reproduces TabulatedPotential::compute's linear interpolation, including the
past-the-end rule, in double precision, and rebuilds mars2's exclusion set
from the EXCLUDE lines plus bond pairs.

Takes the config path as an argument and reads the fixture in place - it
copies nothing and hardcodes no path.

Usage:
    python3 rotor_pair_sum_reference.py <config.bd> [--cutoff 50]
                                        [--no-bond-exclusions] [--no-exclusions]

Compare the printed totals against mars2 run on the same frame with a tiny
timestep (so nothing moves) and the matching terms enabled.
"""
import argparse
import os
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('config', help='path to the .bd config')
ap.add_argument('--cutoff', type=float, default=50.0)
ap.add_argument('--no-bond-exclusions', action='store_true',
                help='do not add bond pairs to the exclusion set')
ap.add_argument('--no-exclusions', action='store_true',
                help='ignore the EXCLUDE file entirely')
args = ap.parse_args()

root = os.path.dirname(os.path.abspath(args.config)) or '.'
bd = open(args.config).read().splitlines()

type_names = [l.split()[1] for l in bd if l.startswith('particle ')]
tidx = {t: i for i, t in enumerate(type_names)}

pair_file = {}
for l in bd:
    if l.startswith('tabulatedFile'):
        i, j, path = l.split()[1].split('@')
        pair_file[(int(i), int(j))] = path
        pair_file[(int(j), int(i))] = path

pos, typ = [], []
particles_file = next(l.split(None, 1)[1].strip() for l in bd if l.startswith('inputParticles'))
for l in open(f'{root}/{particles_file}'):
    p = l.split()
    if len(p) >= 6 and p[0] == 'ATOM':
        typ.append(tidx[p[2]])
        pos.append([float(p[3]), float(p[4]), float(p[5])])
pos = np.asarray(pos, dtype=np.float64)
typ = np.asarray(typ)
print(f'particles: {len(pos)}  types: {len(type_names)}  pair tables: {len(set(pair_file.values()))}')

cache = {}
def table(path):
    if path not in cache:
        d = np.loadtxt(f'{root}/{path}')
        cache[path] = (d[0, 0], d[1, 0] - d[0, 0], np.ascontiguousarray(d[:, 1]))
    return cache[path]

def interp(path, r):
    """Mirror of TabulatedPotential::compute's energy branch."""
    start, step, pot = table(path)
    w = (r - start) / step
    home = np.floor(w).astype(np.int64)
    frac = w - home
    n = len(pot)
    out = np.empty(r.shape, dtype=np.float64)
    past = home >= n - 1
    out[past] = pot[n - 1]
    ok = ~past
    h = np.clip(home[ok], 0, None)
    out[ok] = (pot[h + 1] - pot[h]) * frac[ok] + pot[h]
    return out

excl = set()
excl_file = next((l.split(None, 1)[1].strip() for l in bd if l.startswith('inputExcludes')), None)
for l in (open(f'{root}/{excl_file}') if excl_file else []):
    p = l.split()
    if p and p[0] == 'EXCLUDE' and not args.no_exclusions:
        a, b = int(p[1]), int(p[2])
        excl.add((min(a, b), max(a, b)))
n_excl_file = len(excl)
bonds = []
bonds_file = next((l.split(None, 1)[1].strip() for l in bd if l.startswith('inputBonds')), None)
for l in (open(f'{root}/{bonds_file}') if bonds_file else []):
    p = l.split()
    if p and p[0] == 'BOND':
        a, b = int(p[2]), int(p[3])
        bonds.append((a, b, p[4]))
        if not args.no_bond_exclusions:
            excl.add((min(a, b), max(a, b)))
print(f'exclusions: {n_excl_file} from file + {len(bonds)} bonds = {len(excl)} unique')

CUTOFF = args.cutoff
tree = cKDTree(pos)
pairs = np.asarray(sorted(tree.query_pairs(CUTOFF)), dtype=np.int64)
print(f'pairs within {CUTOFF} A: {len(pairs)}')

if excl:
    keys = pairs[:, 0].astype(np.int64) * len(pos) + pairs[:, 1]
    ex = np.fromiter((a * len(pos) + b for a, b in excl), dtype=np.int64, count=len(excl))
    keep = ~np.isin(keys, ex)
    print(f'  excluded from that set: {(~keep).sum()}')
    pairs = pairs[keep]

d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
ti, tj = typ[pairs[:, 0]], typ[pairs[:, 1]]

nb_energy = 0.0
key = np.minimum(ti, tj).astype(np.int64) * len(type_names) + np.maximum(ti, tj)
for k in np.unique(key):
    m = key == k
    i, j = divmod(int(k), len(type_names))
    nb_energy += interp(pair_file[(i, j)], d[m]).sum()

bond_energy = 0.0
for a, b, path in (bonds if not args.no_bond_exclusions else []):
    r = np.linalg.norm(pos[a] - pos[b])
    bond_energy += float(interp(path, np.array([r]))[0])

print()
print(f'CPU nonbonded energy : {nb_energy:15.2f} kcal/mol')
print(f'CPU bond energy      : {bond_energy:15.2f} kcal/mol')
print(f'CPU bonds + nonbonded: {nb_energy + bond_energy:15.2f} kcal/mol')
