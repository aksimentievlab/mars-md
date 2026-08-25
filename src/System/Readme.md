Folder for stuff related to whole system.

## Difference between SimSystem, SystemState and SimManager:
- SimManager: Main simulation look lives here.
- SystemState: Global state for mutable system states. Gatherings from Patches live here, eventually read by output trajecotory.
- SimSystem: Equivalent to Config in `arbdmodel`. Owns system-wide parameters.
- PatchManager: designed to own and schedule patches across devices.
- PeriodicBox: Each patches holds its own periodic box + a system-wide periodic box.
- RigidBodyManager system is the same level as Patches-- while attached particles on rigid bodies go in Patch, RigidBody lives on its own manager-- Patches are for particles only.
