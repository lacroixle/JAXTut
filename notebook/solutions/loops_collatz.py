# Collatz: how many steps to reach 1? The count depends on n -> unknown in advance
@jax.jit
def collatz_steps(n):
    def _cond(carry):
        x, steps = carry
        return x != 1
        
    def _body(carry):
        x, steps = carry
        x = jnp.where(x % 2 == 0, x // 2, 3*x + 1)
        return x, steps + 1

    _, steps = jax.lax.while_loop(_cond, _body, (n, 0))
    return steps

# pure JAX, so it vmaps over many starting values at once:
collatz_steps_vmap =  jax.vmap(collatz_steps)

n = jnp.arange(1, 1_000_000)
N = collatz_steps_vmap(n)

plt.plot(n, N, ',')
plt.show()