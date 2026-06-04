# ArgoCD Bootstrap README
# ============================================================
## Deployment order

1. Install ArgoCD into the cluster:
   ```bash
   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   ```

2. Apply the AppProject:
   ```bash
   kubectl apply -f k8s/argocd/appproject-tropi-climate.yaml
   ```

3. Deploy the App of Apps (self-manages all subsystem apps):
   ```bash
   kubectl apply -f k8s/argocd/app-of-apps.yaml
   ```

ArgoCD will then auto-sync and deploy all apps in `k8s/argocd/apps/` in sync-wave order:
- Wave 0: monitoring, data-flow (infra prerequisites)
- Wave 1: (reserved for future infra deps)
- Wave 2: analytica, hydrologis, api-gateway (application subsystems)

## Sync policy
- `automated.prune: true` — removes resources deleted from Git
- `automated.selfHeal: true` — reverts manual kubectl edits
- `PruneLast: true` — old resources removed after new ones are healthy
- `CreateNamespace: true` — namespaces created automatically

## Node targeting
All subsystem apps use node affinity + tolerations matching the EKS node groups:
- `analytica` → `dedicated=gpu` nodes
- `hydrologis`, `data-flow` → `dedicated=cpu-ingestion` nodes
- `api-gateway` → `dedicated=cpu-ingestion` nodes
- Batch jobs → `dedicated=spot-batch` nodes

Set these in each workload's pod spec:
```yaml
tolerations:
  - key: dedicated
    value: cpu-ingestion
    effect: NoSchedule
nodeSelector:
  workload: cpu-ingestion
```
