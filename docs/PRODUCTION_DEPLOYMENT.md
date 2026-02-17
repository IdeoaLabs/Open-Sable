"""
Production Deployment Guide for Open-Sable

Complete guide for deploying Open-Sable in production environments.
"""

# Production Deployment Guide

## Overview

This guide covers deploying Open-Sable with:
- Docker containerization
- Kubernetes orchestration
- Load balancing
- Monitoring & logging
- Security hardening
- High availability

---

## 1. Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Create data directory
RUN mkdir -p /data/.opensable

# Set environment
ENV SABLECORE_DATA_DIR=/data/.opensable
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 18789 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:18789/health')"

# Run gateway
CMD ["sable", "gateway"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  gateway:
    build: .
    container_name: opensable-gateway
    ports:
      - "18789:18789"
      - "8000:8000"
    volumes:
      - opensable-data:/data/.opensable
      - ./config:/app/config:ro
    environment:
      - SABLECORE_CONFIG=/app/config/config.yaml
      - OLLAMA_BASE_URL=http://ollama:11434
    networks:
      - opensable-net
    restart: unless-stopped
    depends_on:
      - ollama
      - redis
  
  ollama:
    image: ollama/ollama:latest
    container_name: opensable-ollama
    volumes:
      - ollama-data:/root/.ollama
    ports:
      - "11434:11434"
    networks:
      - opensable-net
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    container_name: opensable-redis
    volumes:
      - redis-data:/data
    networks:
      - opensable-net
    restart: unless-stopped
    command: redis-server --appendonly yes
  
  telegram-bot:
    build: .
    container_name: opensable-telegram
    volumes:
      - opensable-data:/data/.opensable
      - ./config:/app/config:ro
    environment:
      - SABLECORE_CONFIG=/app/config/config.yaml
    networks:
      - opensable-net
    restart: unless-stopped
    command: python -m interfaces.telegram_bot
    depends_on:
      - gateway
  
  discord-bot:
    build: .
    container_name: opensable-discord
    volumes:
      - opensable-data:/data/.opensable
      - ./config:/app/config:ro
    environment:
      - SABLECORE_CONFIG=/app/config/config.yaml
    networks:
      - opensable-net
    restart: unless-stopped
    command: python -m interfaces.discord_bot
    depends_on:
      - gateway
  
  mobile-api:
    build: .
    container_name: opensable-mobile
    ports:
      - "8001:8000"
    volumes:
      - opensable-data:/data/.opensable
      - ./config:/app/config:ro
    environment:
      - SABLECORE_CONFIG=/app/config/config.yaml
    networks:
      - opensable-net
    restart: unless-stopped
    command: python -m interfaces.mobile_api
    depends_on:
      - gateway
  
  prometheus:
    image: prom/prometheus:latest
    container_name: opensable-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - opensable-net
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    container_name: opensable-grafana
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards:ro
    ports:
      - "3000:3000"
    networks:
      - opensable-net
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

volumes:
  opensable-data:
  ollama-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  opensable-net:
    driver: bridge
```

### Build and Run

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f gateway

# Stop all services
docker-compose down
```

---

## 2. Kubernetes Deployment

### namespace.yaml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: opensable
```

### configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opensable-config
  namespace: opensable
data:
  config.yaml: |
    ollama:
      base_url: "http://ollama-service:11434"
    
    gateway:
      host: "0.0.0.0"
      port: 18789
    
    rate_limits:
      user_message_max: 60
      user_message_window: 60
```

### deployment-gateway.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opensable-gateway
  namespace: opensable
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opensable
      component: gateway
  template:
    metadata:
      labels:
        app: opensable
        component: gateway
    spec:
      containers:
      - name: gateway
        image: opensable:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 18789
          name: websocket
        - containerPort: 8000
          name: http
        env:
        - name: SABLECORE_CONFIG
          value: /config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        - name: data
          mountPath: /data/.opensable
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 18789
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 18789
          initialDelaySeconds: 20
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: opensable-config
      - name: data
        persistentVolumeClaim:
          claimName: opensable-data
---
apiVersion: v1
kind: Service
metadata:
  name: opensable-gateway
  namespace: opensable
spec:
  selector:
    app: opensable
    component: gateway
  ports:
  - name: websocket
    port: 18789
    targetPort: 18789
  - name: http
    port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### pvc.yaml

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: opensable-data
  namespace: opensable
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
```

### hpa.yaml (Horizontal Pod Autoscaler)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: opensable-gateway-hpa
  namespace: opensable
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: opensable-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Create config
kubectl apply -f configmap.yaml

# Create PVC
kubectl apply -f pvc.yaml

# Deploy gateway
kubectl apply -f deployment-gateway.yaml

# Deploy autoscaler
kubectl apply -f hpa.yaml

# Check status
kubectl get pods -n opensable
kubectl get svc -n opensable
```

---

## 3. Nginx Reverse Proxy

### nginx.conf

```nginx
upstream opensable_gateway {
    least_conn;
    server gateway1:18789 max_fails=3 fail_timeout=30s;
    server gateway2:18789 max_fails=3 fail_timeout=30s;
    server gateway3:18789 max_fails=3 fail_timeout=30s;
}

upstream opensable_api {
    least_conn;
    server api1:8000 max_fails=3 fail_timeout=30s;
    server api2:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name opensable.example.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name opensable.example.com;
    
    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # WebSocket gateway
    location /ws {
        proxy_pass http://opensable_gateway;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
    
    # HTTP API
    location /api/ {
        proxy_pass http://opensable_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
        add_header Access-Control-Allow-Headers "Authorization, Content-Type";
        
        if ($request_method = OPTIONS) {
            return 204;
        }
    }
    
    # Static files (dashboard)
    location / {
        root /var/www/opensable;
        try_files $uri $uri/ /index.html;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;
    limit_req zone=api_limit burst=50 nodelay;
}
```

---

## 4. Monitoring & Logging

### Prometheus Configuration

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'opensable-gateway'
    static_configs:
      - targets: ['gateway:18789']
    metrics_path: '/metrics'
  
  - job_name: 'opensable-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Grafana Dashboards

Create dashboards for:
- Request rate & latency
- Error rates
- CPU & memory usage
- Active sessions
- Queue length
- Cache hit rate

### Structured Logging

**logging_config.yaml**:
```yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    formatter: json
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: /var/log/opensable/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 10

root:
  level: INFO
  handlers: [console, file]
```

---

## 5. Security Hardening

### Environment Variables

Use secrets management (Kubernetes Secrets, HashiCorp Vault):

```bash
# Create secrets
kubectl create secret generic opensable-secrets \
  --from-literal=telegram-token=YOUR_TOKEN \
  --from-literal=discord-token=YOUR_TOKEN \
  --from-literal=openai-key=YOUR_KEY \
  -n opensable
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: opensable-network-policy
  namespace: opensable
spec:
  podSelector:
    matchLabels:
      app: opensable
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 18789
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: ollama
    ports:
    - protocol: TCP
      port: 11434
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Pod Security Policy

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: opensable-psp
spec:
  privileged: false
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
  - configMap
  - secret
  - persistentVolumeClaim
```

---

## 6. High Availability

### Database Replication

Use PostgreSQL with streaming replication or managed database service.

### Redis Cluster

```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-cluster
  namespace: opensable
spec:
  clusterIP: None
  selector:
    app: redis
  ports:
  - port: 6379
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: opensable
spec:
  serviceName: redis-cluster
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Load Balancer Health Checks

Configure health check endpoints:

```python
# In gateway.py
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # Check dependencies
    ollama_ready = await check_ollama()
    redis_ready = await check_redis()
    
    if ollama_ready and redis_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Not ready")
```

---

## 7. Backup & Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/opensable"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup sessions
tar -czf "$BACKUP_DIR/sessions_$TIMESTAMP.tar.gz" \
    /data/.opensable/sessions/

# Backup analytics
tar -czf "$BACKUP_DIR/analytics_$TIMESTAMP.tar.gz" \
    /data/.opensable/analytics/

# Backup plugins
tar -czf "$BACKUP_DIR/plugins_$TIMESTAMP.tar.gz" \
    /data/.opensable/plugins/

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

1. **Daily automated backups** to S3/GCS
2. **Point-in-time recovery** for databases
3. **Infrastructure as Code** (Terraform/Pulumi)
4. **Runbook documentation** for incidents
5. **Regular recovery drills**

---

## 8. CI/CD Pipeline

### GitHub Actions

```.github/workflows/deploy.yml
name: Deploy Open-Sable

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: myregistry/opensable:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG }}
      - run: |
          kubectl set image deployment/opensable-gateway \
            gateway=myregistry/opensable:${{ github.sha }} \
            -n opensable
```

---

## Production Checklist

- [ ] SSL/TLS certificates configured
- [ ] Secrets management setup (Vault/Secrets Manager)
- [ ] Monitoring & alerting configured (Prometheus/Grafana)
- [ ] Log aggregation setup (ELK/Loki)
- [ ] Backup automation configured
- [ ] Disaster recovery plan documented
- [ ] Load testing completed
- [ ] Security audit performed
- [ ] Rate limiting configured
- [ ] Auto-scaling policies set
- [ ] Health checks implemented
- [ ] Documentation complete
- [ ] Runbooks created
- [ ] On-call rotation established
