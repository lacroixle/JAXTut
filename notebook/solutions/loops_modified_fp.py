@jax.jit(static_argnames=('g', 'max_iter'))
def fixed_point(g, x_init, atol=1e-8, rtol=1e-5, max_iter=200):
    def cond(carry):
        x_prev, x_curr, i = carry
        moving = jnp.logical_not(jnp.isclose(x_prev, x_curr, rtol=rtol, atol=atol))
        return jnp.logical_and(moving, i < max_iter)
    def body(carry):
        _, x_curr, i = carry
        return (x_curr, g(x_curr), i + 1)
    x_prev, x_curr, i = jax.lax.while_loop(cond, body, (x_init, g(x_init), 0))
    return x_curr, i