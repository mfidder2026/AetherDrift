#!/usr/bin/env bash
# Puts a self-contained Python runtime in the app bundle (start-prompt-v2 M6): a relocatable
# CPython (python-build-standalone, via uv) plus every engine dependency in a plain
# site-packages folder. The app then starts the engine without installing anything and
# without internet. Built once per uv.lock and cached in app/build/runtime-cache.
#
# Runs as part of the Xcode build when AETHER_BUNDLE_RUNTIME=1 (build-app.sh sets it for
# Release builds). Debug builds skip it and install the runtime on first launch instead.
set -euo pipefail

if [[ "${AETHER_BUNDLE_RUNTIME:-0}" != "1" ]]; then
  exit 0
fi

RES="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"
ENGINE_SRC="${SRCROOT}/../engine"
UV="${RES}/bin/uv"
LOCK_HASH="$(shasum -a 256 "${ENGINE_SRC}/uv.lock" | cut -c1-12)"
CACHE="${SRCROOT}/build/runtime-cache/${LOCK_HASH}"

if [[ ! -f "${CACHE}/.complete" ]]; then
  echo "Building self-contained runtime for lock ${LOCK_HASH}…"
  STAGE="${CACHE}.partial"
  rm -rf "${STAGE}"
  mkdir -p "${STAGE}"
  UV_PYTHON_INSTALL_DIR="${STAGE}/pythons" "${UV}" python install 3.12 --no-bin >/dev/null
  PY_DIR="$(find "${STAGE}/pythons" -maxdepth 1 -type d -name 'cpython-3.12*' | head -n 1)"
  [[ -n "${PY_DIR}" ]] || { echo "error: no CPython 3.12 installed" >&2; exit 1; }
  mv "${PY_DIR}" "${STAGE}/python"
  rm -rf "${STAGE}/pythons"
  "${UV}" export --frozen --no-dev --no-emit-project --no-hashes \
    --project "${ENGINE_SRC}" -o "${STAGE}/requirements.txt" >/dev/null
  "${UV}" pip install --python "${STAGE}/python/bin/python3.12" --target "${STAGE}/site-packages" \
    --no-compile -r "${STAGE}/requirements.txt" >/dev/null
  # Tests, headers and caches are not needed at runtime.
  find "${STAGE}" -type d \( -name '__pycache__' -o -name 'tests' -o -name 'test' \) -prune -exec rm -rf {} +
  rm -rf "${STAGE}/python/include" "${STAGE}/python/share"
  touch "${STAGE}/.complete"
  rm -rf "${CACHE}"
  mv "${STAGE}" "${CACHE}"
fi

mkdir -p "${RES}/runtime"
rsync -a --delete "${CACHE}/python" "${CACHE}/site-packages" "${RES}/runtime/"
cp "${CACHE}/requirements.txt" "${RES}/runtime/"
echo "${LOCK_HASH}" > "${RES}/runtime/lock-hash"

# Every Mach-O file in the bundle must carry a signature for the app's signature to hold.
IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:--}"
find "${RES}/runtime" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \) -print0 |
  while IFS= read -r -d '' f; do
    case "$(file -b "$f")" in
      *"Mach-O"*"executable"*)
        # Executables (python itself) may load the extension modules signed below.
        codesign --force --sign "${IDENTITY}" --options runtime --timestamp=none \
          --entitlements "${SRCROOT}/AetherDrift/Resources/python.entitlements" "$f" >/dev/null ;;
      *"Mach-O"*)
        codesign --force --sign "${IDENTITY}" --options runtime --timestamp=none "$f" >/dev/null ;;
    esac
  done
