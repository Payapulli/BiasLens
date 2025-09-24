#!/bin/bash

# BiasLens Deployment Script
# This script deploys the BiasLens application to a server

set -e  # Exit on any error

echo "🚀 Starting BiasLens deployment..."

# Configuration
APP_NAME="biaslens"
APP_DIR="/opt/biaslens"
SERVICE_USER="biaslens"
PYTHON_VERSION="3.13"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root. Use sudo when needed."
    exit 1
fi

# Check if we're on a supported OS
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This deployment script is designed for Linux systems."
    exit 1
fi

print_status "Setting up BiasLens application..."

# Create application directory
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Copy application files
print_status "Copying application files..."
cp -r . $APP_DIR/
cd $APP_DIR

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Build FAISS index
print_status "Building FAISS index..."
python scripts/index_docs.py

# Create systemd service file
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/biaslens.service > /dev/null <<EOF
[Unit]
Description=BiasLens API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
ExecStart=$APP_DIR/venv/bin/python app/server_production.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration
print_status "Creating nginx configuration..."
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

    # Static files
    location /static/ {
        alias $APP_DIR/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/biaslens /etc/nginx/sites-enabled/
sudo nginx -t

# Reload systemd and start services
print_status "Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable biaslens
sudo systemctl start biaslens
sudo systemctl reload nginx

# Wait for service to start
sleep 5

# Check if service is running
if systemctl is-active --quiet biaslens; then
    print_status "✅ BiasLens service is running!"
    print_status "🌐 Application is available at: http://$(curl -s ifconfig.me)"
    print_status "📊 API documentation: http://$(curl -s ifconfig.me)/docs"
    print_status "🔍 Health check: http://$(curl -s ifconfig.me)/health"
else
    print_error "❌ BiasLens service failed to start"
    print_error "Check logs with: sudo journalctl -u biaslens -f"
    exit 1
fi

print_status "🎉 Deployment completed successfully!"
print_status "To view logs: sudo journalctl -u biaslens -f"
print_status "To restart: sudo systemctl restart biaslens"
print_status "To stop: sudo systemctl stop biaslens"
