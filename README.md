# N-body Initial Conditions for a stable Galactic Disk
## Code developed for the project. Includes:

### Initial Conditions scripts:
- `plummerIC.py`: Generates a realization of an anisotropic Plummer model
- `mestelIC.py`: Generates a realization of the truncated Mestel disk (see Sellwood 2012). Need to compile Swift with the mestel external potential (+ perturbation if desired)
- `mestelIC_cold_fullmass.py`: Mestel disk without truncation and only circular orbits (to illustrate wakes)

### Visualisation:
- `surfacedensity_multiple.py`: To make images of the Mestel disk
- `film.py`: To make animations of the Mestel disk
- `plotpotential_plummer.py`: Plot plummer density profile against model curve

### Potentials:
Includes source files to implement two new external potentials to SWIFT: Mestel and Mestel with a corotating softened point pass
