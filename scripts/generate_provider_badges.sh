#!/usr/bin/env bash
# Pin each provider's logo to the top-right corner of every bundled avatar.
#
# Reads docs/assets/providers/<provider>.png and writes
# docs/assets/avatars/<provider>/<n>.png (256px message avatars) and
# docs/assets/avatars/36/<provider>/<n>.png (card icons). Requires ImageMagick 7
# (`magick`).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
assets="$root/docs/assets"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# badge <provider> <size> <out>: the logo with a white outline that follows its
# shape, so it reads on any background.
badge() {
  magick "$assets/providers/$1.png" -bordercolor none -border 12 \
    \( +clone -alpha extract -threshold 50% -morphology Dilate Disk:8 \
       -background white -alpha shape \) \
    +swap -compose over -composite -resize "$2x$2" "$3"
}

# stamp <avatar dir> <badge size> <offset>: badge every avatar in the directory.
# The offset keeps the badge inside Slack's rounded corners.
stamp() {
  local dir="$1" size="$2" offset="$3"
  for provider in claude codex; do
    badge "$provider" "$size" "$tmp/$provider-$size.png"
    mkdir -p "$dir/$provider"
    for avatar in "$dir"/*.png; do
      magick "$avatar" "$tmp/$provider-$size.png" -geometry "$offset" -composite \
        -strip "$dir/$provider/$(basename "$avatar")"
    done
  done
}

stamp "$assets/avatars" 76 +174+6
stamp "$assets/avatars/36" 13 +21+2
# Flat art survives a 128-color palette losslessly to the eye, at a third of the size.
for provider in claude codex; do
  magick mogrify -strip -alpha off +dither -colors 128 -define png:color-type=3 \
    "$assets/avatars/$provider"/*.png "$assets/avatars/36/$provider"/*.png
done
echo "badged $(ls "$assets"/avatars/*.png | wc -l | tr -d ' ') avatars per provider"
