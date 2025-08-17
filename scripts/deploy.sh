#!/bin/bash
"""
Deployment Script for MCP and Git Integration

This script provides automated deployment of the MCP and Git integration
system with various deployment modes and configuration options.
"""

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/config"
LOGS_DIR="$PROJECT_DIR/logs"

# Default values
DEPLOYMENT_MODE="development"
COMPOSE_FILE="docker-compose.mcp.yml"
PROFILES=""
BACKUP_ENABLED="false"
MONITORING_ENABLED="false"
HA_ENABLED="false"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
MCP and Git Integration Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy      Deploy the MCP integration system
    stop        Stop all services
    restart     Restart all services
    status      Show status of all services
    logs        Show logs from services
    backup      Create backup of system state
    restore     Restore from backup
    cleanup     Clean up unused resources
    validate    Validate configuration

Options:
    -m, --mode MODE         Deployment mode (development, staging, production)
    -p, --profiles PROFILES Comma-separated list of profiles (backup, monitoring, ha)
    -f, --file FILE         Docker Compose file to use
    -c, --config DIR        Configuration directory
    -h, --help              Show this help message

Examples:
    $0 deploy                                    # Basic deployment
    $0 -m production -p monitoring,backup deploy # Production with monitoring and backup
    $0 -p ha deploy                             # Deploy with high availability
    $0 logs mcp-server                          # Show MCP server logs
    $0 backup                                   # Create system backup

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            -p|--profiles)
                PROFILES="$2"
                shift 2
                ;;
            -f|--file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_DIR="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            deploy|stop|restart|status|logs|backup|restore|cleanup|validate)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Store remaining arguments
    ARGS=("$@")
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed"
        exit 1
    fi

    # Check if we're in a Git repository
    if [ ! -d "$PROJECT_DIR/.git" ]; then
        log_warning "Not in a Git repository. Some features may not work correctly."
    fi

    log_success "Prerequisites validated"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment for $DEPLOYMENT_MODE mode..."

    # Create necessary directories
    mkdir -p "$LOGS_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$PROJECT_DIR/backups"

    # Set environment variables based on deployment mode
    case $DEPLOYMENT_MODE in
        development)
            export LOG_LEVEL="DEBUG"
            export MCP_SERVER_MAX_CONNECTIONS="10"
            export REDIS_MAX_MEMORY="128mb"
            export BACKUP_RETENTION_DAYS="3"
            ;;
        staging)
            export LOG_LEVEL="INFO"
            export MCP_SERVER_MAX_CONNECTIONS="50"
            export REDIS_MAX_MEMORY="256mb"
            export BACKUP_RETENTION_DAYS="7"
            ;;
        production)
            export LOG_LEVEL="WARNING"
            export MCP_SERVER_MAX_CONNECTIONS="100"
            export REDIS_MAX_MEMORY="512mb"
            export BACKUP_RETENTION_DAYS="30"
            ;;
    esac

    # Set profile-based environment variables
    if [[ "$PROFILES" == *"backup"* ]]; then
        BACKUP_ENABLED="true"
        export BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
    fi

    if [[ "$PROFILES" == *"monitoring"* ]]; then
        MONITORING_ENABLED="true"
        export PROMETHEUS_PORT="9090"
        export GRAFANA_PORT="3000"
        export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
    fi

    if [[ "$PROFILES" == *"ha"* ]]; then
        HA_ENABLED="true"
        export REDIS_SENTINEL_PORT="26379"
    fi

    log_success "Environment setup completed"
}

# Validate configuration
validate_configuration() {
    log_info "Validating configuration..."

    # Run configuration validation script
    if [ -f "$SCRIPT_DIR/validate_config.py" ]; then
        python3 "$SCRIPT_DIR/validate_config.py" --config-dir "$CONFIG_DIR"
        if [ $? -ne 0 ]; then
            log_error "Configuration validation failed"
            exit 1
        fi
    else
        log_warning "Configuration validation script not found"
    fi

    log_success "Configuration validated"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."

    cd "$PROJECT_DIR"

    # Build MCP server image
    docker build -f Dockerfile.mcp -t mcp-server:latest .
    if [ $? -ne 0 ]; then
        log_error "Failed to build MCP server image"
        exit 1
    fi

    # Build Git workflow image
    docker build -f Dockerfile.git -t git-workflow:latest .
    if [ $? -ne 0 ]; then
        log_error "Failed to build Git workflow image"
        exit 1
    fi

    # Build backup image if backup is enabled
    if [ "$BACKUP_ENABLED" = "true" ]; then
        docker build -f Dockerfile.backup -t mcp-backup:latest .
        if [ $? -ne 0 ]; then
            log_error "Failed to build backup image"
            exit 1
        fi
    fi

    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying MCP integration services..."

    cd "$PROJECT_DIR"

    # Prepare Docker Compose command
    COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"

    # Add profiles if specified
    if [ -n "$PROFILES" ]; then
        IFS=',' read -ra PROFILE_ARRAY <<< "$PROFILES"
        for profile in "${PROFILE_ARRAY[@]}"; do
            COMPOSE_CMD="$COMPOSE_CMD --profile $profile"
        done
    fi

    # Start services
    $COMPOSE_CMD up -d

    if [ $? -ne 0 ]; then
        log_error "Failed to deploy services"
        exit 1
    fi

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30

    # Check service health
    check_service_health

    log_success "Services deployed successfully"
}

# Check service health
check_service_health() {
    log_info "Checking service health..."

    # Check Redis
    if docker-compose -f "$COMPOSE_FILE" ps redis-mcp | grep -q "Up"; then
        log_success "Redis is running"
    else
        log_error "Redis is not running"
    fi

    # Check MCP Server
    if docker-compose -f "$COMPOSE_FILE" ps mcp-server | grep -q "Up"; then
        log_success "MCP Server is running"
    else
        log_error "MCP Server is not running"
    fi

    # Check Git Workflow
    if docker-compose -f "$COMPOSE_FILE" ps git-workflow | grep -q "Up"; then
        log_success "Git Workflow is running"
    else
        log_error "Git Workflow is not running"
    fi

    # Test health endpoints if available
    if command -v curl &> /dev/null; then
        log_info "Testing health endpoints..."
        
        # Test MCP server health (assuming it exposes health endpoint)
        if curl -f -s http://localhost:8765/health > /dev/null; then
            log_success "MCP Server health endpoint is responding"
        else
            log_warning "MCP Server health endpoint is not responding"
        fi
    fi
}

# Stop services
stop_services() {
    log_info "Stopping MCP integration services..."

    cd "$PROJECT_DIR"

    COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"

    # Add profiles if specified
    if [ -n "$PROFILES" ]; then
        IFS=',' read -ra PROFILE_ARRAY <<< "$PROFILES"
        for profile in "${PROFILE_ARRAY[@]}"; do
            COMPOSE_CMD="$COMPOSE_CMD --profile $profile"
        done
    fi

    $COMPOSE_CMD down

    log_success "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting MCP integration services..."
    stop_services
    deploy_services
}

# Show service status
show_status() {
    log_info "Service status:"

    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" ps
}

# Show logs
show_logs() {
    cd "$PROJECT_DIR"

    if [ ${#ARGS[@]} -gt 0 ]; then
        # Show logs for specific service
        docker-compose -f "$COMPOSE_FILE" logs -f "${ARGS[0]}"
    else
        # Show logs for all services
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Create backup
create_backup() {
    log_info "Creating system backup..."

    # Run backup service
    docker-compose -f "$COMPOSE_FILE" --profile backup run --rm backup python backup_service.py

    log_success "Backup created"
}

# Restore from backup
restore_backup() {
    if [ ${#ARGS[@]} -eq 0 ]; then
        log_error "Please specify backup name to restore"
        exit 1
    fi

    log_info "Restoring from backup: ${ARGS[0]}"

    # This would need to be implemented based on backup service
    log_warning "Restore functionality needs to be implemented"
}

# Cleanup unused resources
cleanup_resources() {
    log_info "Cleaning up unused resources..."

    # Remove unused Docker images
    docker image prune -f

    # Remove unused volumes
    docker volume prune -f

    # Remove unused networks
    docker network prune -f

    log_success "Cleanup completed"
}

# Main function
main() {
    # Parse arguments
    parse_args "$@"

    # Validate command
    if [ -z "$COMMAND" ]; then
        log_error "No command specified"
        show_help
        exit 1
    fi

    # Change to project directory
    cd "$PROJECT_DIR"

    # Execute command
    case $COMMAND in
        deploy)
            validate_prerequisites
            setup_environment
            validate_configuration
            build_images
            deploy_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        backup)
            create_backup
            ;;
        restore)
            restore_backup
            ;;
        cleanup)
            cleanup_resources
            ;;
        validate)
            validate_configuration
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"