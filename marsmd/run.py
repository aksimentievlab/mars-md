"""Command-line entry point: ``python -m marsmd.run CONFIG.bd``.

Parsing, resolution and reporting work today. Launching a simulation is Phase
D and raises with an explanation.
"""

from __future__ import annotations

import argparse
import sys

from .bd import BdError, BdParser
from .bd.apply import BdSimulation, resolve_inputs

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m marsmd.run",
        description="Read an ARBD .bd configuration and run it.",
    )
    p.add_argument("config", metavar="CONFIG.bd", help="configuration file")
    p.add_argument(
        "output", nargs="?", default=None, help="output base name (overrides outputName)"
    )

    # The backend is fixed when the engine is built; this picks a device only.
    p.add_argument("--gpu", type=int, default=0, metavar="N", help="run on device N")

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved configuration and exit without simulating",
    )

    unknown = p.add_mutually_exclusive_group()
    unknown.add_argument(
        "--strict",
        dest="unknown",
        action="store_const",
        const="strict",
        help="raise on an unrecognized keyword (default)",
    )
    unknown.add_argument(
        "--warn",
        dest="unknown",
        action="store_const",
        const="warn",
        help="warn on an unrecognized keyword and continue",
    )
    unknown.add_argument(
        "--ignore",
        dest="unknown",
        action="store_const",
        const="ignore",
        help="silently collect unrecognized keywords",
    )
    p.set_defaults(unknown="strict")

    p.add_argument(
        "--dump-json", metavar="FILE", help="write the parsed config as JSON ('-' for stdout)"
    )
    p.add_argument("--steps", type=int, help="override the configured step count")
    return p


def _summarize(config, resolved, *, load: bool) -> str:
    g = config.globals
    lines = [
        f"config           {config.source_path}",
        f"steps            {g.steps}",
        f"timestep         {g.timestep}",
        f"temperature      {g.temperature}",
        f"cutoff           {g.cutoff}  (pairlist {g.pairlist_cutoff})",
        f"output           {g.output_name}  format={g.output_format or 'default'}",
        f"box              size={g.system_size} origin={g.origin}",
        f"integrators      particle={g.particle_dynamic_type or 'default'} "
        f"rigid_body={g.rigid_body_dynamic_type or 'default'}",
        "",
        f"particle types   {len(config.particles)}",
    ]
    for block in config.particles:
        lines.append(
            f"  {block.name:<12} num={block.num:<8} mass={block.mass:<10} "
            f"grids={len(block.grid_files)}"
        )

    lines.append(f"rigid bodies     {len(config.rigid_bodies)} types")
    for rb in config.rigid_bodies:
        lines.append(
            f"  {rb.name:<12} num={rb.num:<8} mass={rb.mass:<10} "
            f"potential={len(rb.potential_grids)} density={len(rb.density_grids)} "
            f"pmf={len(rb.pmf_grids)}"
        )

    lines += [
        "",
        f"grids referenced {len(resolved.grid_paths)}",
        f"tabulated pairs  {len(config.tabulated_pairs)}",
        f"topology files   {len(config.topology_files)}",
    ]
    if load:
        topo = resolved.topology
        lines += [
            f"  particles      {len(resolved.particles)}",
            f"  bonds          {len(topo.bonds)}",
            f"  angles         {len(topo.angles)}",
            f"  dihedrals      {len(topo.dihedrals)}",
            f"  exclusions     {len(topo.exclusions)}",
            f"  restraints     {len(topo.restraints)}",
            f"  rb coordinates {len(resolved.rigid_body_coords)}",
        ]

    if config.unsupported:
        lines += ["", f"unsupported keys {len(config.unsupported)}"]
        for item in config.unsupported:
            lines.append(f"  line {item.line_no:<5} {item.key:<24} {item.reason}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = BdParser(unknown=args.unknown).parse_file(args.config)
    except BdError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        config.globals.output_name = args.output
    if args.steps is not None:
        config.globals.steps = args.steps

    if args.dump_json:
        payload = config.to_json()
        if args.dump_json == "-":
            print(payload)
        else:
            with open(args.dump_json, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")

    # A dry run must not need the data files to be present.
    try:
        resolved = resolve_inputs(config, load=not args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(_summarize(config, resolved, load=False))
        return 0

    print(_summarize(config, resolved, load=True))

    try:
        sim = BdSimulation(config, gpu=args.gpu, resolved=resolved)
        sim.simulate()
    except (ImportError, NotImplementedError, RuntimeError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
