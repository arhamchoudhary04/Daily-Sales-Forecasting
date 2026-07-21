#!/usr/bin/env bash
# Create a kind cluster, build + load both images, deploy, wait for rollout.
# Images use the same names the manifests reference and are loaded into the node,
# so IfNotPresent finds them with no registry.
set -euo pipefail

CLUSTER="ts-forecast"
BACKEND_IMAGE="ghcr.io/arhamchoudhary04/ts-forecast-backend:latest"
FRONTEND_IMAGE="ghcr.io/arhamchoudhary04/ts-forecast-frontend:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "==> creating kind cluster '${CLUSTER}'"
if kind get clusters 2>/dev/null | grep -qx "${CLUSTER}"; then
  echo "    already exists, reusing"
else
  kind create cluster --config "${ROOT}/k8s/kind-config.yaml"
fi

echo "==> installing ingress-nginx"
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl -n ingress-nginx wait --for=condition=available deploy/ingress-nginx-controller --timeout=180s || \
  echo "    ingress not ready yet, port-forward still works"

echo "==> building images"
docker build -t "${BACKEND_IMAGE}" "${ROOT}/backend"
docker build -t "${FRONTEND_IMAGE}" "${ROOT}/frontend"

echo "==> loading images into kind"
kind load docker-image "${BACKEND_IMAGE}" --name "${CLUSTER}"
kind load docker-image "${FRONTEND_IMAGE}" --name "${CLUSTER}"

echo "==> applying manifests"
# kind-config.yaml is the cluster spec, not a workload manifest
kubectl apply -f "${ROOT}/k8s/namespace.yaml"
kubectl apply \
  -f "${ROOT}/k8s/backend-deployment.yaml" \
  -f "${ROOT}/k8s/backend-service.yaml" \
  -f "${ROOT}/k8s/frontend-deployment.yaml" \
  -f "${ROOT}/k8s/frontend-service.yaml" \
  -f "${ROOT}/k8s/ingress.yaml"

echo "==> waiting for rollout"
kubectl -n forecasting rollout status deploy/backend --timeout=240s
kubectl -n forecasting rollout status deploy/frontend --timeout=240s

echo
kubectl get pods -n forecasting
echo
echo "Reach the app with either:"
echo "  kubectl -n forecasting port-forward svc/frontend 3000:3000   # then http://localhost:3000"
echo "  or add '127.0.0.1 forecast.local' to /etc/hosts and open http://forecast.local"
