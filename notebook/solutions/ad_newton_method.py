def minimize_newton_method(f, x_0, atol=1e-5, rtol=1e-5, maxiter=100):
    H = jax.hessian(f, 0)
    g = jax.grad(f)

    def _step(x):
        return -jnp.linalg.solve(H(x), g(x))

    x = x_0 # Initial position
    s = _step(x) # Initial step
    h = [x] # Position history
    i = 0 # Loop count
    
    # We loop until convergence criterion is satisfied, we're running out of allowed itterations or the function becomes nan.
    while jnp.logical_not(jnp.isclose(jnp.sqrt(jnp.sum(s**2)), 0., atol=atol, rtol=rtol)) and i < maxiter and jnp.logical_not(jnp.any(jnp.isnan(s))):
        x = x + s
        s = _step(x)
        h.append(x+s)
        i += 1

    # Return found position and history
    return x, jnp.array(h)

minimize_newton_method(lambda x: rosenbrock_2d(x, 1., 100.), jnp.array([1.5, 1.5]))