# Draco glTF decoder

Copied unchanged from the installed `three@0.182.0` package at `examples/jsm/libs/draco/gltf/`.

These decoder artifacts are part of [Google Draco](https://github.com/google/draco), distributed under **Apache-2.0**; the license is included as `LICENSE`.

Serve this folder locally and configure Three.js `DRACOLoader` with `setDecoderPath('/models/draco/')`. The WebAssembly loader uses `draco_wasm_wrapper.js` and `draco_decoder.wasm`. The component selects this WebAssembly decoder explicitly; no encoder or unused JavaScript decoder is shipped.
