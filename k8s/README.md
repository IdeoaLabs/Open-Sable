# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying Open-Sable in a production environment.

## Prerequisites

- Kubernetes cluster 1.24+
- kubectl configured
- Helm (optional, for cert-manager)
- Storage class configured (for PersistentVolumeClaims)

## Quick Deploy

```bash
# Create namespace and deploy all components
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n opensable
kubectl get svc -n opensable
```

## Components

### Core Application
- **deployment.yaml**: Open-Sable gateway deployment (2 replicas)
- **namespace.yaml**: Namespace with resource quotas and limits
- **configmap.yaml**: Configuration and secrets
- **ingress.yaml**: NGINX ingress for external access

### Dependencies
- **dependencies.yaml**:
  - Ollama (LLM server)
  - PostgreSQL (database)
  - Redis (cache)

### Monitoring
- **monitoring.yaml**:
  - Jaeger (distributed tracing)
  - Prometheus (metrics)
  - Grafana (dashboards)

### Autoscaling
- **hpa.yaml**: Horizontal Pod Autoscaler (2-10 replicas)

## Configuration

### 1. Update ConfigMap

Edit `k8s/configmap.yaml`:

```yaml
data:
  OLLAMA_BASE_URL: "http://ollama-service:11434"
  DEFAULT_MODEL: "llama3.1:8b"
  # ... other settings
```

### 2. Update Secrets

Edit `k8s/configmap.yaml` (Secret section):

```yaml
stringData:
  TELEGRAM_BOT_TOKEN: "your-actual-token"
  DISCORD_BOT_TOKEN: "your-actual-token"
  OPENAI_API_KEY: "your-api-key"
  # ... other secrets
```

**Production**: Use Kubernetes secrets management or external secret operators:

```bash
# Using kubectl
kubectl create secret generic opensable-secrets \
  --from-literal=TELEGRAM_BOT_TOKEN=your-token \
  --from-literal=DISCORD_BOT_TOKEN=your-token \
  -n opensable

# Or use Sealed Secrets, External Secrets Operator, etc.
```

### 3. Update Ingress

Edit `k8s/ingress.yaml`:

```yaml
spec:
  tls:
  - hosts:
    - your-domain.com  # Change this
    secretName: opensable-tls
  rules:
  - host: your-domain.com  # Change this
```

## Deployment Steps

### 1. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Deploy Dependencies

```bash
kubectl apply -f k8s/dependencies.yaml

# Wait for dependencies to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n opensable --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n opensable --timeout=300s
kubectl wait --for=condition=ready pod -l app=ollama -n opensable --timeout=300s
```

### 3. Deploy Monitoring (Optional)

```bash
kubectl apply -f k8s/monitoring.yaml

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n opensable
# Open http://localhost:3000 (admin/[password from secret])

# Access Jaeger UI
kubectl port-forward svc/jaeger-query 16686:16686 -n opensable
# Open http://localhost:16686
```

### 4. Deploy Open-Sable

```bash
# Apply configuration
kubectl apply -f k8s/configmap.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Enable autoscaling
kubectl apply -f k8s/hpa.yaml
```

### 5. Setup Ingress

```bash
# Install NGINX Ingress Controller (if not already installed)
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace

# Install cert-manager for TLS (if not already installed)
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Apply ingress
kubectl apply -f k8s/ingress.yaml
```

## Verification

### Check Pods

```bash
kubectl get pods -n opensable

# Expected output:
# NAME                         READY   STATUS    RESTARTS   AGE
# opensable-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
# opensable-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
# postgres-xxxxxxxxxx-xxxxx    1/1     Running   0          5m
# redis-xxxxxxxxxx-xxxxx       1/1     Running   0          5m
# ollama-xxxxxxxxxx-xxxxx      1/1     Running   0          5m
```

### Check Services

```bash
kubectl get svc -n opensable

# Expected output:
# NAME                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)
# opensable-gateway   ClusterIP   10.x.x.x        <none>        18789/TCP,9090/TCP
# postgres            ClusterIP   10.x.x.x        <none>        5432/TCP
# redis               ClusterIP   10.x.x.x        <none>        6379/TCP
# ollama-service      ClusterIP   10.x.x.x        <none>        11434/TCP
```

### Check Logs

```bash
# Open-Sable logs
kubectl logs -f deployment/opensable -n opensable

# Specific pod logs
kubectl logs -f opensable-xxxxxxxxxx-xxxxx -n opensable
```

### Test Health Endpoints

```bash
# Port forward to test locally
kubectl port-forward svc/opensable-gateway 18789:18789 -n opensable

# Test health endpoint
curl http://localhost:18789/health

# Test metrics endpoint
curl http://localhost:18789/metrics
```

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment opensable --replicas=5 -n opensable
```

### Autoscaling Status

```bash
# Check HPA status
kubectl get hpa -n opensable

# Detailed HPA info
kubectl describe hpa opensable-hpa -n opensable
```

## Storage

### Persistent Volumes

```bash
# Check PVCs
kubectl get pvc -n opensable

# Expected PVCs:
# - opensable-data (50Gi) - Workflows, images, sessions
# - ollama-models (100Gi) - LLM models
# - postgres-data (20Gi) - Database
# - redis-data (10Gi) - Cache
# - prometheus-data (20Gi) - Metrics
# - grafana-data (10Gi) - Dashboards
```

### Backup

```bash
# Backup Open-Sable data
kubectl exec -it deployment/opensable -n opensable -- tar czf /tmp/backup.tar.gz /data
kubectl cp opensable-xxxxxxxxxx-xxxxx:/tmp/backup.tar.gz ./backup.tar.gz -n opensable

# Backup PostgreSQL
kubectl exec -it deployment/postgres -n opensable -- \
  pg_dump -U opensable opensable > backup.sql
```

## Monitoring

### Prometheus Metrics

```bash
# Port forward Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n opensable

# Open http://localhost:9090
# Query examples:
# - rate(opensable_requests_total[5m])
# - opensable_active_sessions
# - process_cpu_seconds_total
```

### Grafana Dashboards

```bash
# Port forward Grafana
kubectl port-forward svc/grafana 3000:3000 -n opensable

# Open http://localhost:3000
# Default: admin / [password from secret]

# Add Prometheus datasource:
# URL: http://prometheus:9090
```

### Distributed Tracing

```bash
# Port forward Jaeger UI
kubectl port-forward svc/jaeger-query 16686:16686 -n opensable

# Open http://localhost:16686
# Search for traces, view request flows
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod
kubectl describe pod opensable-xxxxxxxxxx-xxxxx -n opensable

# Check events
kubectl get events -n opensable --sort-by='.lastTimestamp'

# Check logs
kubectl logs opensable-xxxxxxxxxx-xxxxx -n opensable --previous
```

### Database Connection Issues

```bash
# Test PostgreSQL connection
kubectl exec -it deployment/postgres -n opensable -- psql -U opensable -c "SELECT 1"

# Check PostgreSQL logs
kubectl logs deployment/postgres -n opensable
```

### Ingress Not Working

```bash
# Check ingress
kubectl describe ingress opensable-ingress -n opensable

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/

# Delete namespace (removes everything)
kubectl delete namespace opensable
```

## Production Checklist

- [ ] Update all secrets with production values
- [ ] Configure proper domain in ingress
- [ ] Setup TLS certificates (cert-manager + Let's Encrypt)
- [ ] Configure resource limits based on load testing
- [ ] Setup backup strategy for PVCs
- [ ] Configure monitoring alerts
- [ ] Setup log aggregation (ELK, Loki)
- [ ] Enable network policies
- [ ] Configure pod security policies
- [ ] Setup disaster recovery plan
- [ ] Document runbooks for common issues

## Advanced Configuration

### Multi-Region Deployment

For high availability across regions, deploy to multiple clusters and use a global load balancer.

### GPU Support (for local image generation)

Add to deployment.yaml:

```yaml
spec:
  template:
    spec:
      containers:
      - name: opensable
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Network Policies

Create network policies to restrict traffic between pods:

```bash
kubectl apply -f k8s/network-policies.yaml
```

## Support

For issues or questions:
- Documentation: `/docs`
- GitHub Issues: https://github.com/yourusername/Open-Sable/issues
- Kubernetes: This README
