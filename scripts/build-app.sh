#!/usr/bin/env bash
# Builds AetherDrift.app (Debug by default; pass "Release" for a release build) and runs the tests.
# Release builds carry their own Python runtime, so the app works offline (bundle-runtime.sh).
# Needs Xcode and XcodeGen (brew install xcodegen).
set -euo pipefail
CONFIG="${1:-Debug}"
cd "$(dirname "$0")/../app"
xcodegen -q
BUNDLE_RUNTIME=0
[[ "$CONFIG" == "Release" ]] && BUNDLE_RUNTIME=1
xcodebuild -project AetherDrift.xcodeproj -scheme AetherDrift -configuration "$CONFIG" \
  -derivedDataPath build -quiet build AETHER_BUNDLE_RUNTIME="$BUNDLE_RUNTIME"
xcodebuild -project AetherDrift.xcodeproj -scheme AetherDrift -derivedDataPath build -quiet test
echo "Klaar: app/build/Build/Products/$CONFIG/AetherDrift.app"
