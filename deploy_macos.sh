#!/bin/bash

# BiasLens macOS Deployment Script
# This script deploys the BiasLens application on macOS

set -e  # Exit on any error

echo "🍎 Starting BiasLens deployment on macOS..."

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

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS systems."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

print_status "Setting up BiasLens application..."

# Create application directory
APP_DIR="$HOME/biaslens"
mkdir -p $APP_DIR

# Copy application files
print_status "Copying application files..."
cp -r . $APP_DIR/
cd $APP_DIR

# Build and run with Docker Compose
print_status "Building Docker containers..."
docker-compose build

print_status "Starting BiasLens services..."
docker-compose up -d

# Wait for services to start
print_status "Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    print_status "✅ BiasLens services are running!"
    
    # Get the local IP
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
    
    print_status "🌐 Application is available at:"
    print_status "   - Frontend: http://localhost/"
    print_status "   - API: http://localhost/analyze"
    print_status "   - API Docs: http://localhost/docs"
    print_status "   - Health Check: http://localhost/health"
    
    if [ ! -z "$LOCAL_IP" ]; then
        print_status "   - Network access: http://$LOCAL_IP/"
    fi
    
    print_status ""
    print_status "📊 To manage the services:"
    print_status "   - View logs: docker-compose logs -f"
    print_status "   - Stop services: docker-compose down"
    print_status "   - Restart services: docker-compose restart"
    print_status "   - View status: docker-compose ps"
    
else
    print_error "❌ Services failed to start"
    print_error "Check logs with: docker-compose logs"
    exit 1
fi

print_status "🎉 Deployment completed successfully!"
print_status "Your BiasLens application is now running in Docker containers!"
