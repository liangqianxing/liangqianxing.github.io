# Flight helmet — homepage sculpture

Source: [KhronosGroup/glTF-Sample-Assets / FlightHelmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/FlightHelmet), retrieved 2026-10-20. Upstream credits Gary Hsu for the Maya conversion (2018, Public).

## License

Model geometry and textures are **CC0-1.0**, as declared in the upstream `LICENSE.md`, copied verbatim alongside this file. Upstream metadata is **CC-BY-4.0**. Logos and trademarks are excluded. See the upstream license and [model README](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/FlightHelmet/README.md).

## Shipped assets

- `FlightHelmet.bust.glb`: 846,976 bytes. Five meshes, 72,032 triangles and fifteen embedded WebP textures. Uses Draco geometry compression and preserves the lens transmission material.
- `poster.png`: a transparent still rendered from the same model and homepage shader. Displays during loading and when JavaScript or WebGL is unavailable.

The original helmet rests on a wooden stand and includes a long breathing hose. This variant removes complete connected components: the wooden base, support column, nameplate, breathing hose, and its nine low metal fittings. It retains the mask, internal head form, side straps and their attached buckles. No cut passes through the face or helmet shell.

Preparation used glTF-Transform 4.x: deduplication, pruning, vertex welding, Draco edgebreaker compression (position 14 bits, normal 10, UV 12), and WebP texture conversion (512 px maximum, quality 80). Geometry and embedded image references were validated by decoding the final GLB. No external asset requests are needed.

Y is up and the face points toward +Z. Final model bounds in original units: X `[-0.151348, 0.162951]`, Y `[0.239642, 0.716000]`, Z `[-0.154152, 0.180071]`. The component centers and scales the model at runtime.

The local WebAssembly Draco decoder in `../draco/` is a separate **Apache-2.0** dependency; its license and source note are included there.
