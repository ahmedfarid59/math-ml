# MathML Parser Web Interface

A comprehensive web interface for the MathML parser with real-time parsing, visual equation rendering, format conversion tools, and shareable mathematical expressions.

## Features

### 🌟 Core Functionality
- **Real-time Mathematical Expression Parsing**: Parse mathematical expressions as you type
- **Multiple Output Format Visualization**: View results in LaTeX, MathML, HTML, ASCII, and SVG formats
- **Domain-Specific Analysis**: Automatic detection and analysis of complex numbers, differential equations, probability distributions, and set theory
- **Interactive Equation Editor**: Syntax highlighting and example expressions
- **Expression Sharing**: Generate shareable links for mathematical expressions
- **Export Capabilities**: Download expressions in various formats
- **Responsive Design**: Works seamlessly on mobile and desktop devices

### 🔧 Technical Features
- **WebSocket Support**: Real-time communication for instant parsing feedback
- **Performance Monitoring**: Track processing time and performance metrics
- **Caching System**: Optimized performance with intelligent caching
- **Collaborative Features**: Share expressions in real-time collaboration rooms
- **RESTful API**: Full API access for integration with other applications

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd mathml_parser/web_interface
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the development server**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Using the Deployment Script

For easier setup and deployment, use the provided deployment script:

```bash
# Quick development setup with all features
python deploy.py --run-dev

# Full development setup with scripts
python deploy.py --env development --install-deps --create-scripts

# Production deployment with Docker
python deploy.py --env production --install-deps --create-docker --all
```

## Usage Guide

### Basic Expression Parsing

1. **Enter Expression**: Type a mathematical expression in the input area
2. **Parse**: Click "Parse Expression" or press Ctrl+Enter
3. **View Results**: See the parsed results in multiple formats
4. **Domain Analysis**: Check for domain-specific features (complex numbers, calculus, etc.)

### Example Expressions

Try these examples to explore different mathematical domains:

#### Complex Numbers
```
3 + 4i
2∠45°
exp(iπ) + 1
```

#### Differential Equations
```
dy/dx = x + y
d²y/dx² + 2dy/dx + y = 0
∂u/∂t = ∂²u/∂x²
```

#### Probability Theory
```
X ~ N(0, 1)
Y ~ Bin(10, 0.3)
P(X > 2) = 0.05
```

#### Set Theory
```
{1, 2, 3, 4}
A ∪ B
{x | x > 0}
```

### Sharing and Exporting

1. **Share Expression**: 
   - Click "Share" button after parsing
   - Copy the generated link to share with others
   - Shared expressions are accessible via direct URLs

2. **Export Expression**:
   - Click "Export" button after parsing
   - Choose from available formats (LaTeX, HTML, MathML, ASCII)
   - Download the expression in your preferred format

## API Reference

### REST API Endpoints

#### Parse Expression
```http
POST /api/parse
Content-Type: application/json

{
  "expression": "3 + 4i"
}
```

**Response:**
```json
{
  "original_expression": "3 + 4i",
  "parsed_result": "...",
  "output_formats": {
    "latex": "3 + 4i",
    "html": "3 + 4<i>i</i>",
    "mathml": "<math>...</math>",
    "ascii": "3 + 4*i"
  },
  "domain_analysis": {
    "complex_numbers": {
      "detected": true,
      "magnitude": 5.0,
      "argument": 53.13
    }
  },
  "processing_time": 0.025,
  "expression_id": "abc123def456"
}
```

#### Get Shared Expression
```http
GET /api/share/{expression_id}
```

#### Export Expression
```http
GET /api/export/{expression_id}/{format}
```

Available formats: `latex`, `html`, `mathml`, `ascii`, `svg`

### WebSocket Events

#### Connect and Parse
```javascript
const socket = io();

// Parse expression in real-time
socket.emit('parse_expression', {
  expression: '∫ x² dx'
});

// Receive results
socket.on('parse_result', (result) => {
  console.log('Parsed result:', result);
});

// Handle errors
socket.on('parse_error', (error) => {
  console.error('Parse error:', error);
});
```

#### Collaboration Features
```javascript
// Join collaboration room
socket.emit('join_collaboration_room', {
  room_id: 'math_study_group'
});

// Share expression with room
socket.emit('share_expression', {
  room_id: 'math_study_group',
  expression: 'x² + y² = r²'
});

// Receive shared expressions
socket.on('expression_shared', (data) => {
  console.log('Expression shared:', data);
});
```

## Configuration

### Environment Variables

Set these environment variables for customization:

```bash
# Flask environment (development/production)
export FLASK_ENV=development

# Secret key for sessions (change in production)
export SECRET_KEY=your-secret-key-here

# Server configuration
export HOST=127.0.0.1
export PORT=5000

# CORS allowed origins (production)
export ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Configuration Files

- `config.py`: Application configuration settings
- `requirements.txt`: Python dependencies
- `app.py`: Main application file

## Deployment

### Development

```bash
# Run with automatic reloading
FLASK_ENV=development python app.py

# Or use the deployment script
python deploy.py --run-dev
```

### Production

#### Option 1: Direct Execution
```bash
FLASK_ENV=production python app.py
```

#### Option 2: Gunicorn (Recommended)
```bash
pip install gunicorn eventlet
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

#### Option 3: Docker
```bash
# Create Docker files
python deploy.py --env production --create-docker

# Build and run
docker-compose up --build -d
```

#### Option 4: Systemd Service (Linux)
```bash
# Create service file
python deploy.py --env production --create-service

# Install service
sudo cp mathml-web.service /etc/systemd/system/
sudo systemctl enable mathml-web
sudo systemctl start mathml-web
```

### Performance Optimization

For production deployment:

1. **Use a reverse proxy** (Nginx) for static files and SSL termination
2. **Enable caching** for frequently accessed expressions
3. **Monitor performance** using the built-in metrics
4. **Scale horizontally** by running multiple instances behind a load balancer

## Architecture

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **MathJax 3**: Mathematical notation rendering
- **Socket.IO**: Real-time communication
- **CodeMirror**: Syntax highlighting (planned)

### Backend
- **Flask**: Web framework
- **Flask-SocketIO**: WebSocket support
- **MathML Parser**: Core mathematical expression processing
- **Performance Optimizer**: Caching and optimization
- **Domain Processors**: Specialized mathematical analysis

### Integration
- **Multi-format Renderer**: Convert between mathematical notations
- **Domain Analysis**: Complex numbers, calculus, probability, set theory
- **Performance Monitoring**: Real-time metrics and optimization
- **Caching System**: LRU, TTL, and adaptive caching strategies

## Browser Compatibility

- **Chrome 90+**: Full support including WebSockets
- **Firefox 88+**: Full support including WebSockets  
- **Safari 14+**: Full support including WebSockets
- **Edge 90+**: Full support including WebSockets
- **Mobile browsers**: Responsive design supports iOS Safari and Chrome Mobile

## Security Considerations

### Development
- Debug mode enabled
- Permissive CORS settings
- Detailed error messages

### Production
- Debug mode disabled
- Restricted CORS origins
- Secure session cookies
- HTTPS enforcement (with reverse proxy)
- Input validation and sanitization

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <pid> /F

# Kill process on port 5000 (Unix/Linux/Mac)
lsof -ti:5000 | xargs kill -9
```

#### Module Import Errors
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:/path/to/mathml_parser"

# Or use the deployment script
python deploy.py --install-deps --run-dev
```

#### WebSocket Connection Issues
- Check firewall settings
- Verify port accessibility
- Ensure eventlet is installed for production

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Issues

Monitor performance metrics in the web interface or check logs for:
- Expression parsing time
- Memory usage
- Cache hit rates
- WebSocket connection counts

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run tests**:
   ```bash
   python -m pytest tests/
   ```
4. **Start development server**:
   ```bash
   python deploy.py --run-dev
   ```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings for all public functions
- Include comprehensive error handling

### Testing

- Write unit tests for new features
- Test WebSocket functionality
- Verify cross-browser compatibility
- Performance testing for complex expressions

## License

This project is part of the MathML Parser suite and follows the same license terms.

## Support

For support and questions:
- Check the troubleshooting section above
- Review the API documentation
- Test with the provided examples
- Verify environment setup using `python deploy.py --env development`

---

**Built with ❤️ for mathematical expression parsing and visualization**