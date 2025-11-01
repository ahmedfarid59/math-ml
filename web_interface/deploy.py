"""
Deployment Scripts and Utilities for MathML Parser Web Interface
===============================================================

This module provides deployment scripts and utilities for running the
MathML parser web interface in different environments.
"""

import os
import sys
import subprocess
import argparse
from typing import List, Optional
import platform
import json


class DeploymentManager:
    """Manages deployment of the web interface."""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.web_interface_root = os.path.dirname(os.path.abspath(__file__))
        self.requirements_file = os.path.join(self.web_interface_root, 'requirements.txt')
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            print(f"✅ Python version {current_version[0]}.{current_version[1]} is compatible")
            return True
        else:
            print(f"❌ Python version {current_version[0]}.{current_version[1]} is too old")
            print(f"   Required: Python {required_version[0]}.{required_version[1]}+")
            return False
    
    def install_dependencies(self, upgrade: bool = False) -> bool:
        """Install required dependencies."""
        print("📦 Installing dependencies...")
        
        try:
            # Check if requirements file exists
            if not os.path.exists(self.requirements_file):
                print(f"❌ Requirements file not found: {self.requirements_file}")
                return False
            
            # Build pip command
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', self.requirements_file]
            if upgrade:
                cmd.append('--upgrade')
            
            # Run pip install
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies:")
                print(f"   {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'flask',
            'flask_socketio',
            'werkzeug'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print("   Run with --install-deps to install them")
            return False
        
        print("✅ All required dependencies are installed")
        return True
    
    def create_run_script(self, environment: str = 'development') -> str:
        """Create platform-specific run script."""
        
        if platform.system() == 'Windows':
            script_name = 'run_web_interface.bat'
            script_content = f'''@echo off
REM MathML Parser Web Interface Runner (Windows)
echo Starting MathML Parser Web Interface...

REM Set environment
set FLASK_ENV={environment}
set PYTHONPATH=%PYTHONPATH%;{self.project_root}

REM Change to web interface directory
cd /d "{self.web_interface_root}"

REM Run the application
python app.py

pause
'''
        else:
            script_name = 'run_web_interface.sh'
            script_content = f'''#!/bin/bash
# MathML Parser Web Interface Runner (Unix/Linux/Mac)
echo "Starting MathML Parser Web Interface..."

# Set environment
export FLASK_ENV={environment}
export PYTHONPATH="$PYTHONPATH:{self.project_root}"

# Change to web interface directory
cd "{self.web_interface_root}"

# Run the application
python app.py
'''
        
        script_path = os.path.join(self.web_interface_root, script_name)
        
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Make executable on Unix systems
            if platform.system() != 'Windows':
                os.chmod(script_path, 0o755)
            
            print(f"✅ Created run script: {script_path}")
            return script_path
            
        except Exception as e:
            print(f"❌ Failed to create run script: {e}")
            return ""
    
    def create_systemd_service(self, user: str = 'www-data', 
                              group: str = 'www-data') -> str:
        """Create systemd service file for Linux deployment."""
        
        service_content = f'''[Unit]
Description=MathML Parser Web Interface
After=network.target

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={self.web_interface_root}
Environment=FLASK_ENV=production
Environment=PYTHONPATH={self.project_root}
ExecStart={sys.executable} app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
        
        service_path = os.path.join(self.web_interface_root, 'mathml-web.service')
        
        try:
            with open(service_path, 'w', encoding='utf-8') as f:
                f.write(service_content)
            
            print(f"✅ Created systemd service file: {service_path}")
            print(f"   To install: sudo cp {service_path} /etc/systemd/system/")
            print(f"   To enable: sudo systemctl enable mathml-web")
            print(f"   To start: sudo systemctl start mathml-web")
            
            return service_path
            
        except Exception as e:
            print(f"❌ Failed to create systemd service: {e}")
            return ""
    
    def create_docker_files(self) -> bool:
        """Create Docker configuration files."""
        
        # Dockerfile
        dockerfile_content = '''FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY web_interface/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Change to web interface directory
WORKDIR /app/web_interface

# Run application
CMD ["python", "app.py"]
'''
        
        # docker-compose.yml
        compose_content = '''version: '3.8'

services:
  mathml-web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-secret-key-here
    volumes:
      - ./logs:/app/web_interface/logs
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - mathml-web
    restart: unless-stopped
'''
        
        # nginx.conf
        nginx_content = '''events {
    worker_connections 1024;
}

http {
    upstream mathml_app {
        server mathml-web:5000;
    }
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        # Proxy to Flask app
        location / {
            proxy_pass http://mathml_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket support
        location /socket.io/ {
            proxy_pass http://mathml_app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Static files
        location /static/ {
            alias /app/web_interface/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
'''
        
        try:
            # Write Dockerfile
            dockerfile_path = os.path.join(self.project_root, 'Dockerfile')
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)
            
            # Write docker-compose.yml
            compose_path = os.path.join(self.project_root, 'docker-compose.yml')
            with open(compose_path, 'w', encoding='utf-8') as f:
                f.write(compose_content)
            
            # Write nginx.conf
            nginx_path = os.path.join(self.project_root, 'nginx.conf')
            with open(nginx_path, 'w', encoding='utf-8') as f:
                f.write(nginx_content)
            
            print("✅ Created Docker configuration files:")
            print(f"   - {dockerfile_path}")
            print(f"   - {compose_path}")
            print(f"   - {nginx_path}")
            print("\n🐳 To build and run with Docker:")
            print("   docker-compose up --build")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create Docker files: {e}")
            return False
    
    def run_development_server(self):
        """Run development server."""
        print("🚀 Starting development server...")
        
        # Set environment
        os.environ['FLASK_ENV'] = 'development'
        os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')};{self.project_root}"
        
        # Change to web interface directory
        os.chdir(self.web_interface_root)
        
        try:
            # Import and run app
            sys.path.insert(0, self.project_root)
            from web_interface.app import main
            return main()
            
        except Exception as e:
            print(f"❌ Failed to start development server: {e}")
            return 1
    
    def deploy(self, environment: str, install_deps: bool = False,
               create_scripts: bool = False, create_docker: bool = False,
               create_service: bool = False) -> bool:
        """Deploy the web interface."""
        
        print(f"🚀 Deploying MathML Parser Web Interface ({environment})")
        print("=" * 60)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Install dependencies if requested
        if install_deps:
            if not self.install_dependencies():
                return False
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Create run scripts if requested
        if create_scripts:
            script_path = self.create_run_script(environment)
            if not script_path:
                return False
        
        # Create systemd service if requested
        if create_service and platform.system() == 'Linux':
            service_path = self.create_systemd_service()
            if not service_path:
                return False
        
        # Create Docker files if requested
        if create_docker:
            if not self.create_docker_files():
                return False
        
        print("\n✅ Deployment preparation complete!")
        
        if environment == 'development':
            print("\n🎯 Next steps:")
            print("   1. Run the development server:")
            if create_scripts:
                if platform.system() == 'Windows':
                    print("      run_web_interface.bat")
                else:
                    print("      ./run_web_interface.sh")
            else:
                print("      python app.py")
            print("   2. Open http://localhost:5000 in your browser")
        
        elif environment == 'production':
            print("\n🎯 Production deployment options:")
            print("   1. Direct execution:")
            print("      FLASK_ENV=production python app.py")
            print("   2. With Gunicorn:")
            print("      gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app")
            if create_docker:
                print("   3. With Docker:")
                print("      docker-compose up --build -d")
            if create_service:
                print("   4. With systemd (Linux):")
                print("      sudo systemctl start mathml-web")
        
        return True


def main():
    """Main function for deployment script."""
    
    parser = argparse.ArgumentParser(
        description='Deploy MathML Parser Web Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Development deployment with dependency installation
  python deploy.py --env development --install-deps --create-scripts
  
  # Production deployment with Docker
  python deploy.py --env production --install-deps --create-docker
  
  # Quick development run
  python deploy.py --run-dev
        '''
    )
    
    parser.add_argument('--env', choices=['development', 'production'], 
                       default='development',
                       help='Deployment environment')
    
    parser.add_argument('--install-deps', action='store_true',
                       help='Install Python dependencies')
    
    parser.add_argument('--create-scripts', action='store_true',
                       help='Create platform-specific run scripts')
    
    parser.add_argument('--create-docker', action='store_true',
                       help='Create Docker configuration files')
    
    parser.add_argument('--create-service', action='store_true',
                       help='Create systemd service file (Linux only)')
    
    parser.add_argument('--run-dev', action='store_true',
                       help='Run development server immediately')
    
    parser.add_argument('--all', action='store_true',
                       help='Create all deployment files')
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = DeploymentManager()
    
    # Handle --run-dev flag
    if args.run_dev:
        # Install dependencies first if needed
        if not manager.check_dependencies():
            print("📦 Installing dependencies first...")
            if not manager.install_dependencies():
                exit(1)
        
        exit(manager.run_development_server())
    
    # Handle --all flag
    if args.all:
        args.create_scripts = True
        args.create_docker = True
        if platform.system() == 'Linux':
            args.create_service = True
    
    # Deploy
    success = manager.deploy(
        environment=args.env,
        install_deps=args.install_deps,
        create_scripts=args.create_scripts,
        create_docker=args.create_docker,
        create_service=args.create_service
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()