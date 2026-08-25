---
marp: true
math: katex
---
<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: "" -->

## A GPU-Accelerated Lattice Boltzmann Solver

#### D2Q9 with Kokkos and MPI

###### Julius Fischer · 226-08-26
HPC with Accelerators

---

# Roadmap: Lattice Boltzmann
- Our goal is a fluid simulation

1. Streaming
- Makes the fluid move
2. Boundary Conditions / Bounce-back & Wrap-Around
- Lets the fluid interact with its container
3. Collision
- 'Adds the phyisics' to the simulation

---
# Discretization
- D2Q9 - 2 Dimensions, 9 Directions/ Channels
- Datastructure: `Kokkos::View` 
- Agnostic to hardware: CPU / GPU

```
typedef Kokkos::View<double***>     DISTRIB;
```

---
# Algorithm: Streaming
<!-- footer: MILESTONE 02–03 -->

- Simple for loop:
    - For every value: *push* it in its direction 
- Contains wrap around logic
    - Branchless Implementation:
     `int new_x = x + x_part(dir);`
     `new_x += (new_x < 0) * size_x;`
---
# Algorithm: Bounce-back


---

## Kokkos-based implementation

<span class="tag">on-node parallelization</span>

<div class="grid2">
<div>

- `BoltzmanLattice` owns state as **Kokkos Views**: `distribution`, `buffer` (double-buffered stream target), `density`, `avg_velocity`
- Every update is a `Kokkos::parallel_for` over an `MDRangePolicy` — same source compiles to OpenMP (CPU) or CUDA (GPU)
- **`collision_fused()`**: density, velocity, $f^{eq}$, BGK relaxation in *one* kernel instead of four — fewer global-memory round-trips

</div>
<div>

<pre class="code-static"><code>Kokkos::parallel_for(
  "CollisionFused", all_nodes_policy(),
  KOKKOS_LAMBDA (const int &x, const int &y) {
    // load f_i, reduce -&gt; rho, u
    // build f_i^eq, relax, write back
    // all in registers, one pass
});</code></pre>

<p class="small">
Build: <code>cmake -B build-cuda -DYALB_ENABLE_CUDA=ON
-DKokkos_ARCH_&lt;GPU&gt;=ON</code>, same <code>src/</code> for CPU and GPU targets.
</p>

</div>
</div>

---

## Correctness by construction

Before trusting performance numbers, the solver has to be *right*.

**Unit tests** (`tests/test_milestone03.cpp`, GoogleTest, run via `ctest`):

| Test | What it checks |
|---|---|
| `MassConservation` | $\sum_i f_i$ summed over the whole lattice is invariant across 100 stream+collide steps on a randomized initial distribution |
| `MomentumConservation` | Total momentum $\sum \mathbf{e}_i f_i$ is likewise invariant |

Both pass to within `1e-5` — the discrete Boltzmann update conserves mass and momentum exactly, as the continuous equations require.

---

## Validation: shear-wave decay

<span class="tag">MILESTONE 04</span>

<div class="grid2">
<div>

- Initialize a sinusoidal velocity perturbation $u_x(y) = \varepsilon \sin(ky)$, $k=2\pi/L$, no forcing, and let it decay (`shear_wave_init()`)
- Analytically, the amplitude decays as $e^{-\nu k^2 t}$ — a direct, independent measurement of the solver's viscosity
- Fit $\ln|u_x|_{max}$ vs. timestep from the simulation: $\nu_{meas} = 0.0558$ vs. $\nu_{analytic} = \frac{1}{3}(\frac{1}{\omega}-\frac{1}{2}) = 0.0556$ at $\omega=1.5$ — **0.4% error**

</div>
<div class="center">

![w:560](img/shear_wave_decay.png)

</div>
</div>

<p class="small">Data from an actual <code>milestone04</code> run (<code>data/04_velocity_slice.csv</code>), not a reference implementation.</p>

---

## Validation: lid-driven cavity

<span class="tag">MILESTONE 05</span>

<div class="grid2">
<div>

- Classic LBM benchmark: no-slip walls on 3 sides (bounce-back), moving lid on top
- $300\times300$ grid, $Re = \dfrac{u_{lid} L}{\nu} \approx 540$, steady state by $t=10{,}000$
- Reproduces the expected topology: **primary vortex** at center + **secondary corner vortex** — only appears once solver & BCs are correct

</div>
<div class="center">

![w:400](img/cavity_steady_state.png)

</div>
</div>

<p class="small">Milestone 05 · <code>bounce_back()</code>: no-slip + moving-lid BCs, <code>src/boltzman.cpp</code></p>

---

## Scaling out: MPI domain decomposition

<span class="tag">MILESTONE 06</span>

<div class="grid2">
<div>

- 2D **Cartesian rank topology**: `MPI_Dims_create` + `MPI_Cart_create`; each rank owns a `size_x/dims[0] × size_y/dims[1]` sub-lattice
- 1-cell **ghost layer** exchanged with all 4 neighbors every step via `MPI_Cart_shift` + `MPI_Sendrecv`
- Boundary slices packed/unpacked by a Kokkos kernel into small staging buffers (a fixed-index slice of the distribution view is non-contiguous under GPU layout)

</div>
<div>

<div class="mpigrid">
<div class="rank">rank 0</div><div class="arrow">↔</div><div class="rank">rank 1</div><div class="arrow">↔</div><div class="rank">rank 2</div>
<div class="arrow">↕</div><div></div><div class="arrow">↕</div><div></div><div class="arrow">↕</div>
<div class="rank">rank 3</div><div class="arrow">↔</div><div class="rank">rank 4</div><div class="arrow">↔</div><div class="rank">rank 5</div>
</div>

<p class="small">1-cell halo exchanged with all 4 neighbors, every step.</p>

<p class="small"><b>One rank per GPU.</b> Staging buffers allocated <i>once</i>, reused every step — an earlier version re-allocated 8 device views per exchange (~160k <code>cudaMalloc</code>/<code>cudaFree</code> pairs over a 10k-step run), capping scaling.</p>

</div>
</div>

---

## Performance & scaling

<span class="tag">MILESTONE 06</span>

- Metric: **MLUPS** — Million Lattice Updates/s $= \dfrac{X \cdot Y \cdot N_{steps}}{t_{run}\times 10^6}$

<div class="center">

| Grid | Timesteps | Runtime | GPUs | **MLUPS** |
|---|---|---|---|---|
| $3000\times3000$ | 10,000 | 8.28 s | 12 (3 nodes × 4 A100) | **495.4** |

</div>

- `scaling_tests.jobs` sweeps **strong scaling** (fixed $3000^2$, GPUs $\in\{1,2,4,8,16\}$) and **weak scaling** (constant $750^2$/GPU, same counts)

<div class="placeholder center">
Scaling sweep queued but not yet completed on the cluster — strong/weak scaling plots go here once scaling_tests.jobs finishes running.
</div>

<p class="small">Expected: near-linear weak scaling (halo is thin relative to a 750×750 subdomain); strong scaling rolls off once per-GPU work drops below what's needed to hide ghost-exchange latency.</p>

---

## Summary & outlook

**What's built:** a D2Q9 Kokkos solver — streaming, BGK collision, bounce-back / moving-lid BCs — validated by conservation tests and shear-wave decay, distributed with MPI Cartesian ghost exchange, one rank per GPU.

**What it does correctly:** mass & momentum conservation to $10^{-5}$; measured viscosity matches shear-wave theory; lid-driven cavity reproduces the expected primary + secondary vortex topology.

**Measured so far:** 495 MLUPS on 12 A100 GPUs at $3000^2$.

**Next:** finish the strong/weak scaling sweep; profile the ghost-exchange vs. compute overlap at high rank counts.

### Questions?
