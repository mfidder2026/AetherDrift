#!/usr/bin/env bash
# Builds AetherDrift.app (Debug by default; pass "Release" for a release build) and runs the tests.
# Needs Xcode and XcodeGen (brew install xcodegen).
set -euo pipefail
CONFIG="${1:-Debug}"
cd "$(dirname "$0")/../app"
xcodegen -q
xcodebuild -project AetherDrift.xcodeproj -scheme AetherDrift -configuration "$CONFIG" \
  -derivedDataPath build -quiet build
xcodebuild -project AetherDrift.xcodeproj -scheme AetherDrift -derivedDataPath build -quiet test
echo "Klaar: app/build/Build/Products/$CONFIG/AetherDrift.app"
