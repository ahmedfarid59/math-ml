"""
Web Interface Configuration and Setup
=====================================

This module provides configuration settings and setup utilities for the
MathML Parser web interface deployment.
"""

import os
from typing import Dict, Any


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Application settings
    APP_NAME = 'MathML Parser Web Interface'
    APP_VERSION = '1.0.0'
    
    # Performance settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    
    # Socket.IO settings
    SOCKETIO_ASYNC_MODE = 'eventlet'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"  # Restrict in production
    
    # Cache settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration."""
        pass


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    TESTING = False
    
    # Database (if needed later)
    DATABASE_URI = 'sqlite:///mathml_dev.db'
    
    # Logging
    LOG_LEVEL = 'DEBUG'
    
    @staticmethod
    def init_app(app):
        """Initialize development environment."""
        Config.init_app(app)
        
        # Development-specific setup
        import logging
        logging.basicConfig(level=logging.DEBUG)


class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or None
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")
    
    # HTTPS settings
    PREFERRED_URL_SCHEME = 'https'
    
    # CORS settings (more restrictive)
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '').split(',')
    
    # Database
    DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///mathml_prod.db'
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    @staticmethod
    def init_app(app):
        """Initialize production environment."""
        Config.init_app(app)
        
        # Production-specific setup
        import logging
        from logging.handlers import RotatingFileHandler
        
        # File logging
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = RotatingFileHandler(
            'logs/mathml_web.log', 
            maxBytes=10240000, 
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('MathML Parser Web Interface startup')


class TestingConfig(Config):
    """Testing configuration."""
    
    DEBUG = True
    TESTING = True
    
    # Test database
    DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Fast cache for testing
    CACHE_DEFAULT_TIMEOUT = 1


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name: str = None) -> Config:
    """
    Get configuration class based on environment.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
                    If None, uses FLASK_ENV environment variable
    
    Returns:
        Configuration class
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])


class WebInterfaceSetup:
    """Utility class for setting up the web interface."""
    
    @staticmethod
    def create_directories():
        """Create necessary directories for the web interface."""
        directories = [
            'templates',
            'static',
            'static/css',
            'static/js',
            'static/images',
            'logs',
            'uploads'  # For potential file upload features
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """
        Validate the environment setup.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            results['errors'].append(
                f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}"
            )
            results['valid'] = False
        else:
            results['info'].append(
                f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
        
        # Check required packages
        required_packages = [
            'flask',
            'flask_socketio',
            'werkzeug'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                results['info'].append(f"Package {package}: ✓ installed")
            except ImportError:
                results['errors'].append(f"Package {package}: ✗ missing")
                results['valid'] = False
        
        # Check optional packages
        optional_packages = [
            'eventlet',
            'gunicorn',
            'whitenoise'
        ]
        
        for package in optional_packages:
            try:
                __import__(package)
                results['info'].append(f"Optional package {package}: ✓ installed")
            except ImportError:
                results['warnings'].append(f"Optional package {package}: ✗ missing")
        
        # Check environment variables
        env_vars = {
            'FLASK_ENV': os.environ.get('FLASK_ENV', 'Not set'),
            'SECRET_KEY': '***' if os.environ.get('SECRET_KEY') else 'Not set',
            'PORT': os.environ.get('PORT', '5000'),
            'HOST': os.environ.get('HOST', '127.0.0.1')
        }
        
        for var, value in env_vars.items():
            results['info'].append(f"Environment {var}: {value}")
        
        # Check file permissions
        try:
            test_file = 'test_write_permission.tmp'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            results['info'].append("File write permissions: ✓ OK")
        except Exception as e:
            results['errors'].append(f"File write permissions: ✗ Error - {e}")
            results['valid'] = False
        
        return results
    
    @staticmethod
    def print_startup_info():
        """Print startup information and validation results."""
        print("=" * 60)
        print("🌐 MathML Parser Web Interface Setup")
        print("=" * 60)
        
        # Validate environment
        validation = WebInterfaceSetup.validate_environment()
        
        if validation['valid']:
            print("✅ Environment validation: PASSED")
        else:
            print("❌ Environment validation: FAILED")
        
        # Print info
        if validation['info']:
            print("\n📋 Information:")
            for info in validation['info']:
                print(f"   {info}")
        
        # Print warnings
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   {warning}")
        
        # Print errors
        if validation['errors']:
            print("\n❌ Errors:")
            for error in validation['errors']:
                print(f"   {error}")
        
        print("\n" + "=" * 60)
        
        if not validation['valid']:
            print("🛑 Setup incomplete. Please fix errors before starting.")
            return False
        
        print("🚀 Setup complete! Ready to start web interface.")
        return True


def create_app(config_name: str = None):
    """
    Application factory function.
    
    Args:
        config_name: Configuration name to use
        
    Returns:
        Configured Flask application
    """
    from flask import Flask
    
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    
    # Initialize configuration
    config_class.init_app(app)
    
    # Create directories
    WebInterfaceSetup.create_directories()
    
    return app


if __name__ == "__main__":
    # Run setup validation
    setup = WebInterfaceSetup()
    
    if setup.print_startup_info():
        print("\n🎯 To start the web interface:")
        print("   python app.py")
        print("\n📦 To install missing dependencies:")
        print("   pip install -r requirements.txt")
    else:
        exit(1)