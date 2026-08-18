def g(x, N):
    return jnp.exp(-jnp.repeat(f(x), N)**2*jnp.linspace(0., 1., N))

N = 10000
x = jax.random.normal(jax.random.key(1337), N)
%timeit df_fwd(lambda x: g(x, N), x).block_until_ready()
%timeit df_rev(lambda x: g(x, N), x).block_until_ready()