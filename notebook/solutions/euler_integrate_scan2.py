def euler_integrate_scan2(f, f0, t):
    h = t[1:]-t[:-1]
    def step(y, x_i):
        h_i = x_i[0]
        t_i = x_i[1]
        return y + h_i * f(t_i, y), y

    return t[1:], jax.lax.scan(step, f0, (h, t[1:]))[1]
