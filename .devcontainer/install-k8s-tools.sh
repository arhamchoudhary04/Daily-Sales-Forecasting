#!/usr/bin/env bash
# Install kubectl, helm and kind for the compose/kind workflow. Runs as
# onCreateCommand (after the container is built), so a hiccup here is only a
# warning, never a recovery-mode container failure.
set -e

echo "installing kubectl..."
curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/v1.30.3/bin/linux/amd64/kubectl"
sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl && rm -f /tmp/kubectl

echo "installing helm..."
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

echo "installing kind..."
curl -fsSLo /tmp/kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
sudo install -m 0755 /tmp/kind /usr/local/bin/kind && rm -f /tmp/kind

echo "k8s tools installed: kubectl / helm / kind"
