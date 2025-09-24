# BiasLens Deployment Guide

This guide covers deploying the BiasLens political bias analysis system to a production server.

## Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Python 3.13+
- Nginx
- Docker (optional)
- At least 2GB RAM
- At least 5GB disk space

## Deployment Options

### Option 1: Direct Deployment (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd BiasLens
   ```

2. **Run the deployment script:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Verify deployment:**
   ```bash
   curl http://your-server-ip/health
   ```

### Option 2: Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **Check status:**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

### Option 3: Manual Deployment

1. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3.13 python3.13-venv nginx
   ```

2. **Set up the application:**
   ```bash
   mkdir -p /opt/biaslens
   cp -r . /opt/biaslens/
   cd /opt/biaslens
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python scripts/index_docs.py
   ```

3. **Create systemd service:**
   ```bash
   sudo tee /etc/systemd/system/biaslens.service > /dev/null <<EOF
   [Unit]
   Description=BiasLens API Server
   After=network.target

   [Service]
   Type=simple
   User=$USER
   WorkingDirectory=/opt/biaslens
   Environment=PATH=/opt/biaslens/venv/bin
   ExecStart=/opt/biaslens/venv/bin/python app/server_production.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   EOF
   ```

4. **Start the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable biaslens
   sudo systemctl start biaslens
   ```

5. **Configure Nginx:**
   ```bash
   sudo tee /etc/nginx/sites-available/biaslens > /dev/null <<EOF
   server {
       listen 80;
       server_name _;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
           proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto \$scheme;
       }
   }
   EOF

   sudo ln -s /etc/nginx/sites-available/biaslens /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

## SSL Configuration (Optional)

To enable HTTPS:

1. **Install Certbot:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   ```

2. **Get SSL certificate:**
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

## Monitoring and Maintenance

### Service Management

```bash
# Check status
sudo systemctl status biaslens

# View logs
sudo journalctl -u biaslens -f

# Restart service
sudo systemctl restart biaslens

# Stop service
sudo systemctl stop biaslens
```

### Health Checks

- **API Health:** `curl http://your-server/health`
- **Frontend:** `curl http://your-server/`
- **API Docs:** `curl http://your-server/docs`

### Logs

- **Application logs:** `sudo journalctl -u biaslens -f`
- **Nginx logs:** `sudo tail -f /var/log/nginx/access.log`
- **Error logs:** `sudo tail -f /var/log/nginx/error.log`

## Performance Tuning

### Nginx Configuration

Edit `/etc/nginx/sites-available/biaslens`:

```nginx
# Add rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    # ... existing config ...
    
    location / {
        limit_req zone=api burst=20 nodelay;
        # ... proxy config ...
    }
}
```

### Application Configuration

Edit `app/server_production.py`:

```python
# Adjust worker processes
uvicorn.run(
    "server_production:app",
    host="0.0.0.0",
    port=8000,
    workers=4,  # Adjust based on CPU cores
    # ... other config ...
)
```

## Troubleshooting

### Common Issues

1. **Service won't start:**
   ```bash
   sudo journalctl -u biaslens -f
   ```

2. **Port already in use:**
   ```bash
   sudo lsof -i :8000
   sudo kill -9 <PID>
   ```

3. **Permission issues:**
   ```bash
   sudo chown -R $USER:$USER /opt/biaslens
   ```

4. **Nginx configuration errors:**
   ```bash
   sudo nginx -t
   ```

### Performance Issues

1. **High memory usage:**
   - Reduce `top_k` in retrieval
   - Implement caching
   - Use smaller models

2. **Slow responses:**
   - Enable gzip compression
   - Add caching headers
   - Optimize database queries

## Security Considerations

1. **Firewall:**
   ```bash
   sudo ufw allow 22
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw enable
   ```

2. **Rate limiting:** Already configured in nginx.conf

3. **Input validation:** Implemented in the API

4. **HTTPS:** Use Let's Encrypt for SSL certificates

## Backup and Recovery

### Backup

```bash
# Backup application
tar -czf biaslens-backup-$(date +%Y%m%d).tar.gz /opt/biaslens

# Backup data
cp -r /opt/biaslens/data /backup/biaslens-data-$(date +%Y%m%d)
```

### Recovery

```bash
# Restore application
tar -xzf biaslens-backup-YYYYMMDD.tar.gz -C /

# Restart services
sudo systemctl restart biaslens
sudo systemctl reload nginx
```

## Updates

1. **Stop service:**
   ```bash
   sudo systemctl stop biaslens
   ```

2. **Update code:**
   ```bash
   cd /opt/biaslens
   git pull
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Restart service:**
   ```bash
   sudo systemctl start biaslens
   ```

## Support

For issues and questions:
- Check logs: `sudo journalctl -u biaslens -f`
- API documentation: `http://your-server/docs`
- Health check: `http://your-server/health`
