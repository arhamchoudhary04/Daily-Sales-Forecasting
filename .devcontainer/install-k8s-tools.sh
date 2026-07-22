#!/usr/bin/env bash
# Install kubectl, helm and kind (Codespaces / devcontainer). Best-effort:
# a network hiccup here shouldn't fail container creation.
set -e

echo "installing kubectl..."
curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/v1.30.3/bin/linux/amd64/kubectl"
sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl && rm -f /tmp/kubectl

echo "installing helm..."
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

echo "installing kind..."
curl -fsSLo /tmp/kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
sudo install -m 0755 /tmp/kind /usr/local/bin/kind && rm -f /tmp/kind

echo "done: $(kubectl version --client --output=yaml >/dev/null 2>&1 && echo kubectl) $(helm version --short 2>/dev/null) kind $(kind version 2>/dev/null)"
