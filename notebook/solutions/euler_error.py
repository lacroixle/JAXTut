def euler_error(h):
    t, y = euler_integrate_scan(A, 1., 0, 1., h)
    return jnp.max(jnp.abs(y-jnp.exp(t)))

h = jnp.logspace(-5, -1, 20)
errors = jnp.array([euler_error(h_i) for h_i in h])

# The following will not work:
#errors = jax.vmap(euler_error)(s)

plt.plot(h, errors, label="$\\max{|y_\\mathrm{euler}-y_\\mathrm{analytical}}|$")
plt.plot(h, h, label="$\\propto h$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$h$")
plt.ylabel("Residuals")
plt.legend()
plt.grid()
plt.show()