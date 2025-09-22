#!/bin/bash

# Deployment script for ASR API
set -e

# Configuration
REPO_URL="https://github.com/silaschalwe/farmvoiceassistant_backend.git"
APP_DIR="/opt/asr-api"
DOCKER_IMAGE="asr-api:latest"
CONTAINER_NAME="asr-api-container"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to check if Git repo has updates
check_for_updates() {
    cd $APP_DIR
    git fetch origin
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse "@{u}")
    
    if [ $LOCAL = $REMOTE ]; then
        log "No updates available"
        return 1
    else
        log "Updates found!"
        return 0
    fi
}

# Function to deploy application
deploy() {
    log "Starting deployment..."
    
    # Create app directory if it doesn't exist
    if [ ! -d "$APP_DIR" ]; then
        log "Creating application directory..."
        sudo mkdir -p $APP_DIR
        sudo chown $USER:$USER $APP_DIR
        
        log "Cloning repository..."
        git clone $REPO_URL $APP_DIR
    else
        log "Updating repository..."
        cd $APP_DIR
        git pull origin main
    fi
    
    cd $APP_DIR
    
    # Stop existing container
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Build new image
    log "Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    
    # Run new container
    log "Starting new container..."
    docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p 5000:5000 \
        -v $(pwd)/logs:/app/logs \
        -e HOST=0.0.0.0 \
        -e PORT=5000 \
        -e DEBUG=false \
        $DOCKER_IMAGE
    
    # Wait for container to be healthy
    log "Waiting for application to be ready..."
    for i in {1..30}; do
        if curl -f http://localhost:5000/health >/dev/null 2>&1; then
            log "Application is healthy!"
            break
        fi
        if [ $i -eq 30 ]; then
            error "Application failed to start properly"
            exit 1
        fi
        sleep 2
    done
    
    log "Deployment completed successfully!"
}

# Function to setup auto-update service
setup_auto_update() {
    log "Setting up auto-update service..."
    
    # Create update script
    cat > /tmp/asr-api-update.sh << 'EOF'
#!/bin/bash
cd /opt/asr-api
if git fetch origin && [ $(git rev-parse HEAD) != $(git rev-parse @{u}) ]; then
    echo "Updates found, deploying..."
    /opt/asr-api/deploy.sh
else
    echo "No updates available"
fi
EOF
    
    sudo mv /tmp/asr-api-update.sh /usr/local/bin/asr-api-update.sh
    sudo chmod +x /usr/local/bin/asr-api-update.sh
    
    # Create systemd timer for auto-updates
    sudo tee /etc/systemd/system/asr-api-update.timer > /dev/null << EOF
[Unit]
Description=ASR API Auto Update Timer
Requires=asr-api-update.service

[Timer]
OnCalendar=*:0/15  # Check every 15 minutes
Persistent=true

[Install]
WantedBy=timers.target
EOF

    sudo tee /etc/systemd/system/asr-api-update.service > /dev/null << EOF
[Unit]
Description=ASR API Auto Update Service
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/asr-api-update.sh
User=$USER
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable asr-api-update.timer
    sudo systemctl start asr-api-update.timer
    
    log "Auto-update service configured!"
}

# Main execution
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "update")
        if check_for_updates; then
            deploy
        fi
        ;;
    "setup-auto-update")
        setup_auto_update
        ;;
    "status")
        if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
            log "Container is running"
            docker logs --tail 10 $CONTAINER_NAME
        else
            warn "Container is not running"
        fi
        ;;
    *)
        echo "Usage: $0 {deploy|update|setup-auto-update|status}"
        exit 1
        ;;
esac