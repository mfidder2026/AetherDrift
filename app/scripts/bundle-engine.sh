#!/usr/bin/env bash
# Xcode build phase: copies the engine source and the uv binary into the app bundle.
# The app installs the Python runtime from these on first launch (start-prompt-v2 §8.13).
set -euo pipefail

RES="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"
ENGINE_SRC="${SRCROOT}/../engine"

mkdir -p "${RES}/engine" "${RES}/bin"
rsync -a --delete --delete-excluded --exclude '__pycache__' --exclude '*.pyc' \
  "${ENGINE_SRC}/aether" "${RES}/engine/"
cp "${ENGINE_SRC}/pyproject.toml" "${ENGINE_SRC}/uv.lock" "${RES}/engine/"

UV="$(command -v uv || true)"
for candidate in /opt/homebrew/bin/uv "${HOME}/.local/bin/uv" /usr/local/bin/uv; do
  [[ -z "${UV}" && -x "${candidate}" ]] && UV="${candidate}"
done
if [[ -z "${UV}" ]]; then
  echo "error: uv not found; install it with 'brew install uv'" >&2
  exit 1
fi
cp -L "${UV}" "${RES}/bin/uv"
chmod 755 "${RES}/bin/uv"
# Re-sign the nested binary so the bundle signature stays valid.
codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY:--}" --options runtime "${RES}/bin/uv" >/dev/null
