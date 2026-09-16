# Scientific scope

## Target quantity

The method infers a three-dimensional, symmetric positive-definite correlation tensor `Q` from sparse oriented two-dimensional sections.

For a unit 3D direction `v`,

```text
ell(v)^(-2) = v^T Q v
```

where `ell(v)` is the characteristic directional correlation length.

For section `s`, let `B_s` be a 3 x 2 orthonormal matrix whose columns span the section. An in-plane unit vector `u` maps to the ambient direction

```text
v = B_s u
```

and the section observes the restriction

```text
Q_s = B_s^T Q B_s.
```

This is the geometric mechanism by which measurements from differently oriented sections are combined.

## Directional correlation measurement

For a binary pore indicator `chi` (1 = pore, 0 = solid), the code centers the field by its pore fraction and evaluates a finite-domain autocorrelation with zero-padding and explicit overlap normalization. Directional curves are sampled by interpolation through the 2D autocorrelation field.

The default characteristic length is the first lag at which the normalized autocorrelation crosses `exp(-1)`; the crossing location is linearly interpolated between neighboring lags. Directions that do not cross within the permitted lag window are treated as censored and excluded from tensor fitting.

## Linear inverse problem

Each valid observation contributes one row

```text
[vx^2, vy^2, vz^2, 2 vx vy, 2 vx vz, 2 vy vz]
```

and a target `1/ell^2`. A general symmetric 3 x 3 tensor has six independent components, so the aggregate design matrix must have rank six.

A single plane can provide rank at most 3. Two complementary planes provide rank 5 in the ideal orthogonal construction. Three appropriately oriented planes can provide rank 6. Full rank is necessary but not sufficient: the condition number is also reported because poor section geometry amplifies measurement noise.

## What is not inferred

The method does not uniquely recover:

- voxel-by-voxel 3D pore geometry;
- pore topology or connectivity;
- higher-order correlation statistics;
- permeability, elastic, electrical, or thermal tensors.

Those quantities require additional measurements, assumptions, or property-specific models. The tensor produced here is a second-order structural descriptor of directional pore-space correlation.
