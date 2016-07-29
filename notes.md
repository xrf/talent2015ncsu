  - VMC: accept near 100% step size too small (and vice versa)
  - DMC: keep branch ratio low (~10%)

time step:

    ⟨V, dT⟩ ≪ 1
    ⟨T dT⟩ ≪ 1

make sure `E_T` is not too far off otherwise branching ratio would be > 10

keep statistics on `⟨T⟩`, `⟨V⟩`, and `⟨r^2⟩`

write out (small) "blocks" (i.e. sample groups/bins) for later re-analysis

write out the state of the walkers after a run

Jastro function must go to a constant (`f' -> 0`) at the `L / 2` boundary

    u(r) -> u(r) + u(L - r) - 2 u(L/2)
    u(r) -> 0 for r -> L / 2

---

personal idea (!): maybe we could use AD to calculate gradients?
