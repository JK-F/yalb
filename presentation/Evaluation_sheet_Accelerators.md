| Evaluation sheet for ""High-Performance Computing: Distributed-memory parallelization on GPUs and accelerators""
| Summer term 2026
| Language of the presentation:  English
| Project type: Lattice Boltzmann
| Contents of the simulation code|||
| (Evaluation points: 0=missing, 1=partial or wrong, 2=complete and correct)|||
| The Contents criteria below (rows 13-22) apply only to Lattice Boltzmann projects. For any other project type, set 'Project type' (above) to 'Other project', leave rows 13-22 blank, and enter a single continuous content score (0-20) in the hard-gate section below - the maximum is unchanged, so the overall scale is identical. Framework note: on the taught Kokkos + MPI track, MPI (Milestone 06) is mandatory; for frameworks not taught in class, MPI is not required - mark the four MPI rows and the Kokkos parallel_for row N/A by leaving their points blank.
| Criterion | Points| Comment|
| Streaming operator correctly implemented (Milestone 02)| 2
| Collision operator with BGK approximation implemented (Milestone 03)| 2||
| Density and velocity field computation from distribution function| 2| |
| Shear-wave decay: viscosity measurement matches analytical prediction (Milestone 04)| 2||
| Bounce-back boundary conditions implemented (Milestone 05) | 2 
| Lid-driven cavity: steady-state velocity field shown (Milestone 05)| 2
| MPI domain decomposition with ghost cell exchange (Milestone 06)| 2||
| Validation: MPI results match serial implementation| 2||
| Strong scaling plot with absolute performance numbers (MLUPS) and discussion of parallel efficiency (Milestone 06)| 2||
| Kokkos parallel_for used for on-node parallelization| 2||
| Quality of the final presentation||
| (Evaluation points: 0=large deficiencies, 1=acceptable, 2=perfect)
| Criterion | Points | Comment|
| Overall structure of the presentation           | 2      |            |
| Clarity of explanations          | 2      |            |
| Clarity of figures and visualizations           | 2      |            |
| Live demo or convincing results shown           | 2      |            |
| Understanding of the physics (responses to questions)          | 2      |            |
| Understanding of the numerics and parallelization (responses to questions)    | 2      |            |
|    |        |            |
| Code quality      |        |            |
| (Evaluation points: 0=large deficiencies, 1=acceptable, 2=perfect)            |        |            |
|    |        |            |
| Criterion         | Points | Comment    |
| Code compiles (Release and Debug builds)        | 2      |            |
| Code organization and readability| 2      |            |
| Proper use of CMake build system | 2      |            |
| Reproducibility*  | 2      |            |
|    |        |            |
| *can the code be compiled and run with the provided instructions              |        |            |
|    |        |            |
| Failure criteria and penalties (hard gates)     |        |            |
| (These gates override the point-based grade below. Set the grader inputs in column B.)       |        |            |
|    |        |            |
| Criterion         | Input  | Comment / guidance        |
| Parallelization framework used   | Kokkos + MPI (taught) | Choose the track. 'Other framework (not taught)' exempts the student from the MPI requirement (higher entry barrier).              |
| Student can explain their own code (answers questions on how it works)        | Yes    | FAILURE CRITERION: 'No' = automatic fail (final grade 5.0), regardless of points.     |
| Sum of all points | 40     |            |
| Potential maximum # points       | 40     |            |
|    |        |            |
| Grade             | 1.000  |            |
| Rounded grade     | 1.0    |            |
