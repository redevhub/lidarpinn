"""
Parameterised shear constraint (generalised power law with learned exponent).

Where Ekman constrains the momentum balance (a second-order ODE coupling u and
v through Coriolis), this constrains the shear profile directly: the wind speed
is required to follow a power law in height whose exponent is inferred per
sample from the surface state,

    U(z) = U_ref (z / z_ref)^alpha

which in differential form, and therefore imposable pointwise on the decoder
without reference to any particular reference height, reads

    z dU/dz - alpha U = 0.

Physical basis: the power law is the standard engineering description of the
surface-layer wind profile, with alpha a function of stability and roughness
(Hellman/Justus). Unlike Ekman it says nothing about direction, so the
constraint is applied to the speed only and the direction is left to the data.
That is precisely why it is a useful contrast: it tests whether any physical
shear constraint helps, or whether the null result is specific to Ekman.

The exponent is a per-sample latent, like u_star and 1/L in the Ekman
formulation, produced by the existing physics head. Only the first derivative
of the decoder is needed.

Typical alpha: 0.1 (unstable/rough) to 0.4 (stable). Clamped to [0.0, 0.6] to
keep the constraint inside the range where the power law is a description of
real profiles rather than an arbitrary curve.
"""
import tensorflow as tf

ALPHA_MIN, ALPHA_MAX = 0.0, 0.6


def shear_residual(model, ctx, alpha, z_bot, z_top, uv_scale,
                   nquad=24):
    """Mean squared residual of  z dU/dz - alpha U = 0  over the column.

    model.profile(ctx, z) must return (B, K, 2); the constraint is applied to
    the speed |(u,v)|. alpha: (B,1), clamped internally. Returns a scalar.
    """
    B = tf.shape(ctx)[0]
    s = tf.linspace(0.0, 1.0, nquad)
    z_nodes = z_bot + (z_top - z_bot) * s
    z = tf.tile(z_nodes[None, :], [B, 1])                    # (B,Q)

    a = tf.clip_by_value(alpha, ALPHA_MIN, ALPHA_MAX)        # (B,1)

    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(z)
        uv = model.profile(ctx, z)                           # (B,Q,2)
        U = tf.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2 + 1e-8)
    dU = t.gradient(U, z)                                    # (B,Q)
    del t

    # z dU/dz - alpha U = 0, normalised by the speed scale so the magnitude is
    # comparable to the Ekman residual it replaces
    res = (z * dU - a * U) / uv_scale
    return tf.reduce_mean(tf.square(res))
