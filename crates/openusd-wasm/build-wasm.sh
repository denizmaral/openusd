#!/bin/bash
# Builds the viewer's USD module: crates/openusd-wasm/pkg/openusd.{js,d.ts} and openusd_bg.wasm.
#
# The size profile is set here rather than in the workspace manifest, so the fork's changes stay inside
# this crate and upstream syncs do not conflict. The wasm-bindgen CLI must match the wasm-bindgen crate
# in Cargo.lock exactly; a matching one is installed under target/ so a global CLI pinned for another
# project is left alone.

set -euo pipefail
cd "$(dirname "$0")/../.."

export CARGO_PROFILE_RELEASE_OPT_LEVEL=z
export CARGO_PROFILE_RELEASE_LTO=true
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
export CARGO_PROFILE_RELEASE_PANIC=abort

cargo build --release --target wasm32-unknown-unknown -p openusd-wasm

VERSION=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' Cargo.lock)
CLI="target/wasm-bindgen-cli-$VERSION/bin/wasm-bindgen"
if [ ! -x "$CLI" ]; then
  cargo install wasm-bindgen-cli --version "$VERSION" --locked --root "target/wasm-bindgen-cli-$VERSION"
fi

"$CLI" --target web --out-name openusd --out-dir crates/openusd-wasm/pkg \
  target/wasm32-unknown-unknown/release/openusd_wasm.wasm
# Optional: wasm-opt shrinks the module by roughly a tenth more.
if command -v wasm-opt >/dev/null; then
  wasm-opt -Oz --strip-debug --strip-producers -o crates/openusd-wasm/pkg/openusd_bg.wasm crates/openusd-wasm/pkg/openusd_bg.wasm
fi
echo "Built crates/openusd-wasm/pkg/openusd_bg.wasm ($(du -h crates/openusd-wasm/pkg/openusd_bg.wasm | cut -f1))"
