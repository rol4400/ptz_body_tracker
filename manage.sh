#!/bin/bash
# PTZ Camera Tracker - Docker Installation and Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="ptz-camera-tracker"
IMAGE_NAME="ptz-tracker"
LOG_FILE="./logs/install.log"

# Create logs directory
mkdir -p logs

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "For WSL2: https://docs.docker.com/desktop/wsl/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Function to build the container
build_container() {
    print_status "Building PTZ Camera Tracker container..."
    
    docker-compose build ptz-tracker
    
    if [ $? -eq 0 ]; then
        print_success "Container built successfully"
    else
        print_error "Failed to build container"
        exit 1
    fi
}

# Function to start the service
start_service() {
    print_status "Starting PTZ Camera Tracker service..."
    
    docker-compose up -d ptz-tracker
    
    if [ $? -eq 0 ]; then
        print_success "Service started successfully"
        print_status "Container logs: docker-compose logs -f ptz-tracker"
    else
        print_error "Failed to start service"
        exit 1
    fi
}

# Function to stop the service
stop_service() {
    print_status "Stopping PTZ Camera Tracker service..."
    
    docker-compose down
    
    if [ $? -eq 0 ]; then
        print_success "Service stopped successfully"
    else
        print_error "Failed to stop service"
        exit 1
    fi
}

# Function to restart the service
restart_service() {
    print_status "Restarting PTZ Camera Tracker service..."
    
    stop_service
    sleep 2
    start_service
}

# Function to show service status
show_status() {
    print_status "PTZ Camera Tracker Service Status:"
    
    docker-compose ps
    
    echo ""
    print_status "Resource Usage:"
    docker stats "$CONTAINER_NAME" --no-stream 2>/dev/null || print_warning "Container not running"
    
    echo ""
    print_status "Health Status:"
    docker inspect "$CONTAINER_NAME" --format='{{.State.Health.Status}}' 2>/dev/null || print_warning "Container not found"
}

# Function to show logs
show_logs() {
    print_status "Showing PTZ Camera Tracker logs..."
    docker-compose logs -f ptz-tracker
}

# Function to enter container shell
enter_shell() {
    print_status "Entering container shell..."
    docker-compose exec ptz-tracker /bin/bash
}

# Function to update the service
update_service() {
    print_status "Updating PTZ Camera Tracker service..."
    
    # Pull latest changes (if using git)
    if [ -d ".git" ]; then
        git pull
    fi
    
    # Rebuild and restart
    build_container
    restart_service
    
    print_success "Service updated successfully"
}

# Function to install as system service (systemd)
install_systemd_service() {
    print_status "Installing as systemd service..."
    
    # Create systemd service file
    cat > /tmp/ptz-tracker.service << EOF
[Unit]
Description=PTZ Camera Tracker
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker-compose up -d ptz-tracker
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    # Install the service
    sudo mv /tmp/ptz-tracker.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable ptz-tracker.service
    
    print_success "Systemd service installed"
    print_status "Use 'sudo systemctl start ptz-tracker' to start"
    print_status "Use 'sudo systemctl status ptz-tracker' to check status"
}

# Function to test OSC connectivity
test_osc() {
    print_status "Testing OSC connectivity..."
    
    # Install python3-osc if not available
    python3 -c "
import time
from pythonosc import udp_client

client = udp_client.SimpleUDPClient('127.0.0.1', 8081)

print('Testing OSC commands...')
print('Sending start command...')
client.send_message('/ptz/start', 1)
time.sleep(2)

print('Requesting status...')
client.send_message('/ptz/status', 1)
time.sleep(1)

print('Sending stop command...')
client.send_message('/ptz/stop', 1)

print('OSC test completed. Check container logs for responses.')
" 2>/dev/null || print_warning "python-osc not available for testing"
}

# Function to show help
show_help() {
    echo "PTZ Camera Tracker - Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install      - Check dependencies and build container"
    echo "  start        - Start the PTZ tracker service"
    echo "  stop         - Stop the PTZ tracker service"
    echo "  restart      - Restart the PTZ tracker service"
    echo "  status       - Show service status and resource usage"
    echo "  logs         - Show and follow service logs"
    echo "  shell        - Enter container shell for debugging"
    echo "  update       - Update and restart the service"
    echo "  test-osc     - Test OSC connectivity"
    echo "  systemd      - Install as systemd service"
    echo "  dev          - Start in development mode with GUI"
    echo "  help         - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install       # Initial setup"
    echo "  $0 start         # Start service"
    echo "  $0 logs          # Monitor logs"
    echo "  $0 test-osc      # Test OSC commands"
}

# Main script logic
case "$1" in
    install)
        check_docker
        build_container
        print_success "Installation completed"
        print_status "Run '$0 start' to start the service"
        ;;
    start)
        check_docker
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    shell)
        enter_shell
        ;;
    update)
        check_docker
        update_service
        ;;
    test-osc)
        test_osc
        ;;
    systemd)
        install_systemd_service
        ;;
    dev)
        print_status "Starting in development mode..."
        docker-compose --profile dev up ptz-tracker-dev
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac