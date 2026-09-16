# Applying the method to your own 2D rock sections

1. Segment each 2D image into pore and solid.
2. Record the physical pixel size.
3. Record the orientation of the two image axes in one common 3D coordinate system.
4. Express those axes as the columns of a 3 x 2 orthonormal basis matrix.
5. Create a JSON manifest following `sections_manifest.example.json`.
6. Run `ssa-infer manifest.json --output result.json`.

At least three suitably complementary section orientations are required for a general rank-six 3D tensor. Merely supplying three images does not guarantee identifiability; the command reports the actual global design-matrix rank and condition number.

The output is a 3D **correlation tensor**, not a synthesized 3D pore volume.
