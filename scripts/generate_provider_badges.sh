#!/usr/bin/env bash
# Pin each provider's logo to the top-right corner of every 64px card avatar.
#
# Reads docs/assets/providers/<provider>.png and writes
# docs/assets/avatars/64/<provider>/<n>.png. Requires ImageMagick 7 (`magick`).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
assets="$root/docs/assets"
badge_size=22
# Top-right, a few pixels in so the badge clears the face and survives a tight crop.
offset="+39+3"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for provider in claude codex; do
  badge="$tmp/$provider.png"
  # White outline that follows the logo's shape, so it reads on any background.
  magick "$assets/providers/$provider.png" -bordercolor none -border 12 \
    \( +clone -alpha extract -threshold 50% -morphology Dilate Disk:8 \
       -background white -alpha shape \) \
    +swap -compose over -composite -resize "${badge_size}x${badge_size}" "$badge"
  out="$assets/avatars/64/$provider"
  mkdir -p "$out"
  for avatar in "$assets"/avatars/64/*.png; do
    magick "$avatar" "$badge" -geometry "$offset" -composite -strip "$out/$(basename "$avatar")"
  done
done
echo "badged $(ls "$assets"/avatars/64/*.png | wc -l | tr -d ' ') avatars per provider"
