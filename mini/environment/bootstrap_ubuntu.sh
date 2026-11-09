#!/usr/bin/env bash
# Run as root in a NEW project WSL distro. Re-running preserves local data.
set -euo pipefail
if [[ $(id -u) != 0 ]]; then
  echo 'Run with wsl -d GrepSeekUbuntu -u root -- bash <this script>' >&2
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
  ripgrep grep gawk sed coreutils findutils git rsync ca-certificates curl bubblewrap util-linux
if ! id grepseek >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash grepseek
fi
install -d -o grepseek -g grepseek /home/grepseek/grepseek
echo 'Ubuntu packages ready; project user: grepseek (no password or sudo grant).'
