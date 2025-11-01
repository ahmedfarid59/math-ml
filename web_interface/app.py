"""
Interactive Web Interface for MathML Parser
===========================================

This module provides a comprehensive web interface for the MathML parser
with real-time parsing, visual equation rendering, format conversion tools,
and shareable mathematical expressions.

Features:
- Real-time mathematical expression parsing and rendering
- Multiple output format visualization (LaTeX, MathML, ASCII, etc.)
- Interactive equation editor with syntax highlighting
- Mathematical domain-specific tools (complex numbers, calculus, etc.)
- Shareable expression permalinks
- Export capabilities for various formats
- Responsive design for mobile and desktop
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import hashlib
import time
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid
import base64
from urllib.parse import quote, unquote

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import MathML parser components
    from mathml_parser.core.parser import MathMLParser
    from mathml_parser.core.multi_format import MultiFormatRenderer
    from mathml_parser.domains import (
        ComplexNumberProcessor, DifferentialEquationProcessor,
        ProbabilityProcessor, SetTheoryProcessor
    )
    from mathml_parser.performance import PerformanceOptimizer, cached, monitor
except ImportError as e:
    print(f"Warning: Could not import MathML parser components: {e}")
    # Create mock classes for development
    class MathMLParser:
        def parse(self, expression): return f"Parsed: {expression}"
    
    class MultiFormatRenderer:
        def render_all(self, parsed): return {"latex": parsed, "html": parsed}
    
    class ComplexNumberProcessor:
        def parse_complex(self, expr): return None
    
    class DifferentialEquationProcessor:
        def parse_ode(self, expr): return None
    
    class ProbabilityProcessor:
        def parse_distribution(self, expr): return None
    
    class SetTheoryProcessor:
        def parse_set(self, expr): return None
    
    class PerformanceOptimizer:
        def __init__(self): pass
        def cached_operation(self): return lambda x: x
        def monitor_performance(self, name): return lambda x: x


@dataclass
class ExpressionResult:
    """Container for parsed expression results."""
    original_expression: str
    parsed_result: Any
    output_formats: Dict[str, str]
    domain_analysis: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: float = 0.0
    expression_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MathExpressionProcessor:
    """
    Core processor for mathematical expressions with domain-specific analysis.
    """
    
    def __init__(self):
        """Initialize the expression processor."""
        self.parser = MathMLParser()
        self.renderer = MultiFormatRenderer()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Domain processors
        self.complex_processor = ComplexNumberProcessor()
        self.de_processor = DifferentialEquationProcessor()
        self.prob_processor = ProbabilityProcessor()
        self.set_processor = SetTheoryProcessor()
        
        # Expression cache for sharing
        self.expression_cache = {}
    
    @cached("expression_cache")
    @monitor("parse_expression")
    def process_expression(self, expression: str) -> ExpressionResult:
        """
        Process mathematical expression with comprehensive analysis.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            ExpressionResult with all analysis data
        """
        start_time = time.perf_counter()
        
        try:
            # Parse expression
            parsed_result = self.parser.parse(expression)
            
            # Generate multiple output formats
            output_formats = self.renderer.render_all(parsed_result)
            
            # Domain-specific analysis
            domain_analysis = self._analyze_domains(expression)
            
            processing_time = time.perf_counter() - start_time
            
            # Generate unique ID for sharing
            expression_id = self._generate_expression_id(expression)
            
            result = ExpressionResult(
                original_expression=expression,
                parsed_result=str(parsed_result),
                output_formats=output_formats,
                domain_analysis=domain_analysis,
                processing_time=processing_time,
                timestamp=time.time(),
                expression_id=expression_id
            )
            
            # Cache for sharing
            self.expression_cache[expression_id] = result
            
            return result
            
        except Exception as e:
            return ExpressionResult(
                original_expression=expression,
                parsed_result=None,
                output_formats={},
                domain_analysis={},
                error_message=str(e),
                processing_time=time.perf_counter() - start_time,
                timestamp=time.time()
            )
    
    def _analyze_domains(self, expression: str) -> Dict[str, Any]:
        """Analyze expression across different mathematical domains."""
        analysis = {}
        
        # Complex numbers
        complex_result = self.complex_processor.parse_complex(expression)
        if complex_result:
            analysis['complex_numbers'] = {
                'detected': True,
                'magnitude': getattr(complex_result, 'magnitude', None),
                'argument': getattr(complex_result, 'argument_degrees', None),
                'form': getattr(complex_result, 'form', 'unknown')
            }
        
        # Differential equations
        ode_result = self.de_processor.parse_ode(expression)
        if ode_result:
            analysis['differential_equations'] = {
                'detected': True,
                'type': ode_result.equation_type.value,
                'order': ode_result.order,
                'linearity': ode_result.linearity
            }
        
        # Probability distributions
        prob_result = self.prob_processor.parse_distribution(expression)
        if prob_result:
            analysis['probability'] = {
                'detected': True,
                'distribution_type': prob_result.distribution_type.value,
                'parameters': prob_result.parameters,
                'properties': prob_result.properties
            }
        
        # Set theory
        set_result = self.set_processor.parse_set(expression)
        if set_result:
            analysis['set_theory'] = {
                'detected': True,
                'cardinality': set_result.cardinality,
                'is_finite': set_result.is_finite,
                'description': set_result.description
            }
        
        return analysis
    
    def _generate_expression_id(self, expression: str) -> str:
        """Generate unique ID for expression sharing."""
        hash_object = hashlib.md5(expression.encode())
        return hash_object.hexdigest()[:12]
    
    def get_cached_expression(self, expression_id: str) -> Optional[ExpressionResult]:
        """Retrieve cached expression by ID."""
        return self.expression_cache.get(expression_id)


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize processor
processor = MathExpressionProcessor()

# Session management
active_sessions = {}


@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')


@app.route('/api/parse', methods=['POST'])
def api_parse():
    """API endpoint for parsing mathematical expressions."""
    try:
        data = request.get_json()
        expression = data.get('expression', '').strip()
        
        if not expression:
            return jsonify({'error': 'No expression provided'}), 400
        
        # Process expression
        result = processor.process_expression(expression)
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/share/<expression_id>')
def api_share(expression_id):
    """API endpoint for retrieving shared expressions."""
    try:
        result = processor.get_cached_expression(expression_id)
        
        if not result:
            return jsonify({'error': 'Expression not found'}), 404
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/share/<expression_id>')
def share_page(expression_id):
    """Web page for shared expressions."""
    result = processor.get_cached_expression(expression_id)
    
    if not result:
        return render_template('error.html', 
                             error_message="Expression not found"), 404
    
    return render_template('share.html', result=result)


@app.route('/api/formats/<expression_id>')
def api_formats(expression_id):
    """API endpoint for getting all output formats of an expression."""
    try:
        result = processor.get_cached_expression(expression_id)
        
        if not result:
            return jsonify({'error': 'Expression not found'}), 404
        
        return jsonify({
            'expression_id': expression_id,
            'original_expression': result.original_expression,
            'formats': result.output_formats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<expression_id>/<format_type>')
def api_export(expression_id, format_type):
    """API endpoint for exporting expressions in specific formats."""
    try:
        result = processor.get_cached_expression(expression_id)
        
        if not result:
            return jsonify({'error': 'Expression not found'}), 404
        
        if format_type not in result.output_formats:
            return jsonify({'error': f'Format {format_type} not available'}), 400
        
        content = result.output_formats[format_type]
        
        # Determine content type and filename extension
        content_types = {
            'latex': 'text/plain',
            'html': 'text/html',
            'mathml': 'application/mathml+xml',
            'ascii': 'text/plain',
            'svg': 'image/svg+xml'
        }
        
        extensions = {
            'latex': 'tex',
            'html': 'html',
            'mathml': 'mml',
            'ascii': 'txt',
            'svg': 'svg'
        }
        
        content_type = content_types.get(format_type, 'text/plain')
        extension = extensions.get(format_type, 'txt')
        filename = f"expression_{expression_id}.{extension}"
        
        response = app.response_class(
            content,
            mimetype=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def api_history():
    """API endpoint for getting expression history."""
    try:
        # In a real application, this would be user-specific
        # For demo, return recent expressions from cache
        recent_expressions = list(processor.expression_cache.values())[-10:]
        
        history = []
        for result in recent_expressions:
            history.append({
                'expression_id': result.expression_id,
                'original_expression': result.original_expression,
                'timestamp': result.timestamp,
                'processing_time': result.processing_time
            })
        
        return jsonify({'history': history})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# WebSocket events for real-time features
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    active_sessions[session_id] = {
        'connected_at': time.time(),
        'expressions_processed': 0
    }
    
    emit('connected', {'session_id': session_id})
    print(f"Client connected: {session_id}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    session_id = session.get('session_id')
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
        print(f"Client disconnected: {session_id}")


@socketio.on('parse_expression')
def handle_parse_expression(data):
    """Handle real-time expression parsing."""
    try:
        expression = data.get('expression', '').strip()
        session_id = session.get('session_id')
        
        if not expression:
            emit('parse_error', {'error': 'No expression provided'})
            return
        
        # Process expression
        result = processor.process_expression(expression)
        
        # Update session statistics
        if session_id in active_sessions:
            active_sessions[session_id]['expressions_processed'] += 1
        
        # Emit result
        emit('parse_result', result.to_dict())
        
        # Broadcast to other users in the same room (for collaborative features)
        # emit('expression_shared', {
        #     'expression': expression,
        #     'session_id': session_id
        # }, broadcast=True, include_self=False)
        
    except Exception as e:
        emit('parse_error', {'error': str(e)})


@socketio.on('join_collaboration_room')
def handle_join_room(data):
    """Handle joining collaborative rooms."""
    room_id = data.get('room_id')
    if room_id:
        join_room(room_id)
        emit('joined_room', {'room_id': room_id})
        emit('user_joined', {
            'session_id': session.get('session_id'),
            'timestamp': time.time()
        }, room=room_id, include_self=False)


@socketio.on('leave_collaboration_room')
def handle_leave_room(data):
    """Handle leaving collaborative rooms."""
    room_id = data.get('room_id')
    if room_id:
        leave_room(room_id)
        emit('left_room', {'room_id': room_id})
        emit('user_left', {
            'session_id': session.get('session_id'),
            'timestamp': time.time()
        }, room=room_id)


@socketio.on('share_expression')
def handle_share_expression(data):
    """Handle sharing expressions in collaborative rooms."""
    room_id = data.get('room_id')
    expression = data.get('expression')
    
    if room_id and expression:
        # Process expression
        result = processor.process_expression(expression)
        
        # Share with room
        emit('expression_shared', {
            'expression': expression,
            'result': result.to_dict(),
            'shared_by': session.get('session_id'),
            'timestamp': time.time()
        }, room=room_id, include_self=False)


def create_templates():
    """Create HTML templates for the web interface."""
    
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create static directory
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
    
    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}MathML Parser Web Interface{% endblock %}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- MathJax -->
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <!-- CodeMirror for syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.0/codemirror.min.css">
    <!-- Custom CSS -->
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">
    
    {% block extra_head %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                🧮 MathML Parser
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('index') }}">Parser</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" onclick="showHistory()">History</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" onclick="showHelp()">Help</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% block content %}{% endblock %}
    </main>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Socket.IO -->
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <!-- CodeMirror -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.0/codemirror.min.js"></script>
    <!-- Custom JS -->
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
    
    {% block extra_scripts %}{% endblock %}
</body>
</html>'''
    
    # Main page template
    index_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <!-- Input Section -->
    <div class="col-lg-6">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">Mathematical Expression Input</h5>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <label for="expression-input" class="form-label">Enter your mathematical expression:</label>
                    <textarea class="form-control" id="expression-input" rows="4" 
                              placeholder="Enter mathematical expressions like: x^2 + 3*sin(x), 2 + 3i, dy/dx = x + y, etc."></textarea>
                </div>
                
                <div class="row">
                    <div class="col-sm-6">
                        <button type="button" class="btn btn-primary w-100" onclick="parseExpression()">
                            Parse Expression
                        </button>
                    </div>
                    <div class="col-sm-6">
                        <button type="button" class="btn btn-outline-secondary w-100" onclick="clearInput()">
                            Clear
                        </button>
                    </div>
                </div>
                
                <!-- Domain Examples -->
                <div class="mt-3">
                    <h6>Example Expressions:</h6>
                    <div class="btn-group-vertical w-100" role="group">
                        <button type="button" class="btn btn-outline-info btn-sm" onclick="setExample('3 + 4i')">
                            Complex Numbers: 3 + 4i
                        </button>
                        <button type="button" class="btn btn-outline-info btn-sm" onclick="setExample('dy/dx = x + y')">
                            Differential Eq: dy/dx = x + y
                        </button>
                        <button type="button" class="btn btn-outline-info btn-sm" onclick="setExample('X ~ N(0, 1)')">
                            Probability: X ~ N(0, 1)
                        </button>
                        <button type="button" class="btn btn-outline-info btn-sm" onclick="setExample('{1, 2, 3, 4}')">
                            Set Theory: {1, 2, 3, 4}
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Performance Info -->
        <div class="card mt-3">
            <div class="card-header">
                <h6 class="card-title mb-0">Performance Metrics</h6>
            </div>
            <div class="card-body">
                <div id="performance-info">
                    <p class="text-muted">Parse an expression to see performance metrics.</p>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Output Section -->
    <div class="col-lg-6">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">Results</h5>
                <div class="btn-group" role="group">
                    <button type="button" class="btn btn-outline-primary btn-sm" onclick="shareExpression()" id="share-btn" disabled>
                        Share
                    </button>
                    <button type="button" class="btn btn-outline-success btn-sm" onclick="exportExpression()" id="export-btn" disabled>
                        Export
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div id="results-container">
                    <p class="text-muted">Enter a mathematical expression to see the parsed results.</p>
                </div>
            </div>
        </div>
        
        <!-- Domain Analysis -->
        <div class="card mt-3">
            <div class="card-header">
                <h6 class="card-title mb-0">Domain Analysis</h6>
            </div>
            <div class="card-body">
                <div id="domain-analysis">
                    <p class="text-muted">Domain-specific analysis will appear here.</p>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Status indicator -->
<div class="fixed-bottom p-3">
    <div class="text-end">
        <span id="connection-status" class="badge bg-secondary">Connecting...</span>
    </div>
</div>

<!-- Share Modal -->
<div class="modal fade" id="shareModal" tabindex="-1">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Share Expression</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <p>Share this link to let others view your expression:</p>
                <div class="input-group">
                    <input type="text" class="form-control" id="share-link" readonly>
                    <button class="btn btn-outline-secondary" type="button" onclick="copyShareLink()">
                        Copy
                    </button>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Export Modal -->
<div class="modal fade" id="exportModal" tabindex="-1">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Export Expression</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <p>Choose format to export:</p>
                <div class="list-group">
                    <a href="#" class="list-group-item list-group-item-action" onclick="downloadFormat('latex')">
                        LaTeX (.tex)
                    </a>
                    <a href="#" class="list-group-item list-group-item-action" onclick="downloadFormat('html')">
                        HTML (.html)
                    </a>
                    <a href="#" class="list-group-item list-group-item-action" onclick="downloadFormat('mathml')">
                        MathML (.mml)
                    </a>
                    <a href="#" class="list-group-item list-group-item-action" onclick="downloadFormat('ascii')">
                        ASCII Text (.txt)
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Share page template
    share_template = '''{% extends "base.html" %}

{% block title %}Shared Expression - MathML Parser{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-lg-8">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">Shared Mathematical Expression</h5>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <label class="form-label"><strong>Original Expression:</strong></label>
                    <div class="bg-light p-3 rounded">
                        <code>{{ result.original_expression }}</code>
                    </div>
                </div>
                
                {% if result.output_formats %}
                <div class="mb-3">
                    <label class="form-label"><strong>Rendered Output:</strong></label>
                    {% if result.output_formats.html %}
                    <div class="bg-light p-3 rounded">
                        {{ result.output_formats.html|safe }}
                    </div>
                    {% endif %}
                </div>
                
                <!-- Format tabs -->
                <ul class="nav nav-tabs" id="formatTabs" role="tablist">
                    {% for format_name in result.output_formats.keys() %}
                    <li class="nav-item" role="presentation">
                        <button class="nav-link {% if loop.first %}active{% endif %}" 
                                id="{{ format_name }}-tab" data-bs-toggle="tab" 
                                data-bs-target="#{{ format_name }}" type="button" role="tab">
                            {{ format_name.title() }}
                        </button>
                    </li>
                    {% endfor %}
                </ul>
                
                <div class="tab-content" id="formatTabsContent">
                    {% for format_name, format_content in result.output_formats.items() %}
                    <div class="tab-pane fade {% if loop.first %}show active{% endif %}" 
                         id="{{ format_name }}" role="tabpanel">
                        <pre class="bg-light p-3 mt-3"><code>{{ format_content }}</code></pre>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if result.domain_analysis %}
                <div class="mt-4">
                    <h6>Domain Analysis:</h6>
                    <div class="bg-light p-3 rounded">
                        <pre>{{ result.domain_analysis|tojson(indent=2) }}</pre>
                    </div>
                </div>
                {% endif %}
                
                <div class="mt-4">
                    <small class="text-muted">
                        Processed in {{ "%.3f"|format(result.processing_time) }}s
                        • Generated {{ result.timestamp|round|int }}
                        • ID: {{ result.expression_id }}
                    </small>
                </div>
            </div>
        </div>
        
        <div class="text-center mt-3">
            <a href="{{ url_for('index') }}" class="btn btn-primary">
                Parse Your Own Expression
            </a>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Error page template
    error_template = '''{% extends "base.html" %}

{% block title %}Error - MathML Parser{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="text-center">
            <h1 class="display-1">❌</h1>
            <h2>Oops! Something went wrong</h2>
            <p class="text-muted">{{ error_message }}</p>
            <a href="{{ url_for('index') }}" class="btn btn-primary">
                Go Back to Parser
            </a>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Write templates
    with open(os.path.join(templates_dir, 'base.html'), 'w', encoding='utf-8') as f:
        f.write(base_template)
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_template)
    
    with open(os.path.join(templates_dir, 'share.html'), 'w', encoding='utf-8') as f:
        f.write(share_template)
    
    with open(os.path.join(templates_dir, 'error.html'), 'w', encoding='utf-8') as f:
        f.write(error_template)
    
    # Create CSS file
    css_content = '''/* Custom styles for MathML Parser Web Interface */

body {
    background-color: #f8f9fa;
}

.card {
    border: none;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: box-shadow 0.3s ease;
}

.card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

#expression-input {
    font-family: 'Courier New', monospace;
    border: 2px solid #e9ecef;
    transition: border-color 0.3s ease;
}

#expression-input:focus {
    border-color: #0d6efd;
    box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25);
}

.btn-group-vertical .btn {
    text-align: left;
    margin-bottom: 2px;
}

#results-container {
    min-height: 200px;
}

.math-output {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 0.5rem 0;
}

.domain-badge {
    display: inline-block;
    margin: 0.25rem;
    padding: 0.375rem 0.75rem;
    background-color: #e7f3ff;
    border: 1px solid #b6d7ff;
    border-radius: 0.375rem;
    font-size: 0.875rem;
}

.performance-metric {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid #e9ecef;
}

.performance-metric:last-child {
    border-bottom: none;
}

.performance-value {
    font-weight: bold;
    color: #0d6efd;
}

.error-message {
    color: #dc3545;
    background-color: #f8d7da;
    border: 1px solid #f1aeb5;
    border-radius: 0.375rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}

.success-message {
    color: #155724;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.375rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}

#connection-status {
    cursor: pointer;
    transition: all 0.3s ease;
}

.format-tab-content {
    max-height: 300px;
    overflow-y: auto;
}

.navbar-brand {
    font-weight: bold;
}

.fixed-bottom {
    pointer-events: none;
}

.fixed-bottom span {
    pointer-events: all;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .container {
        padding-left: 10px;
        padding-right: 10px;
    }
    
    .btn-group-vertical .btn {
        font-size: 0.875rem;
    }
    
    #results-container {
        min-height: 150px;
    }
}

/* Animation for loading states */
.loading {
    position: relative;
    color: transparent !important;
}

.loading::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    width: 20px;
    height: 20px;
    margin: -10px 0 0 -10px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #0d6efd;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Syntax highlighting for mathematical expressions */
.math-expression {
    font-family: 'Courier New', monospace;
    background-color: #f8f9fa;
    padding: 0.5rem;
    border-radius: 0.25rem;
    border: 1px solid #dee2e6;
}

/* Toast notifications */
.toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1055;
}'''
    
    with open(os.path.join(static_dir, 'css', 'style.css'), 'w', encoding='utf-8') as f:
        f.write(css_content)
    
    # Create JavaScript file
    js_content = '''// MathML Parser Web Interface JavaScript

class MathMLWebInterface {
    constructor() {
        this.socket = io();
        this.currentExpressionId = null;
        this.initializeEventHandlers();
        this.initializeSocketHandlers();
    }

    initializeEventHandlers() {
        // Enter key in textarea
        document.getElementById('expression-input').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.parseExpression();
            }
        });

        // Auto-resize textarea
        const textarea = document.getElementById('expression-input');
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        });
    }

    initializeSocketHandlers() {
        this.socket.on('connected', (data) => {
            console.log('Connected to server:', data.session_id);
            this.updateConnectionStatus('Connected', 'success');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus('Disconnected', 'danger');
        });

        this.socket.on('parse_result', (result) => {
            this.displayResults(result);
            this.hideLoading();
        });

        this.socket.on('parse_error', (error) => {
            this.displayError(error.error);
            this.hideLoading();
        });

        // Connection status
        this.socket.on('connect', () => {
            this.updateConnectionStatus('Connected', 'success');
        });

        this.socket.on('connect_error', () => {
            this.updateConnectionStatus('Connection Error', 'danger');
        });
    }

    parseExpression() {
        const expression = document.getElementById('expression-input').value.trim();
        
        if (!expression) {
            this.displayError('Please enter a mathematical expression');
            return;
        }

        this.showLoading();
        this.socket.emit('parse_expression', { expression: expression });
    }

    displayResults(result) {
        const container = document.getElementById('results-container');
        
        if (result.error_message) {
            this.displayError(result.error_message);
            return;
        }

        this.currentExpressionId = result.expression_id;

        // Enable share and export buttons
        document.getElementById('share-btn').disabled = false;
        document.getElementById('export-btn').disabled = false;

        let html = '';

        // Original expression
        html += `<div class="mb-3">
            <label class="form-label"><strong>Original Expression:</strong></label>
            <div class="math-expression">${this.escapeHtml(result.original_expression)}</div>
        </div>`;

        // Rendered output
        if (result.output_formats && Object.keys(result.output_formats).length > 0) {
            html += `<div class="mb-3">
                <label class="form-label"><strong>Rendered Output:</strong></label>`;
            
            // Create tabs for different formats
            const formats = Object.keys(result.output_formats);
            
            html += `<ul class="nav nav-tabs" id="outputTabs" role="tablist">`;
            formats.forEach((format, index) => {
                const active = index === 0 ? 'active' : '';
                html += `<li class="nav-item" role="presentation">
                    <button class="nav-link ${active}" id="${format}-tab" data-bs-toggle="tab" 
                            data-bs-target="#${format}" type="button" role="tab">
                        ${format.charAt(0).toUpperCase() + format.slice(1)}
                    </button>
                </li>`;
            });
            html += `</ul>`;

            html += `<div class="tab-content" id="outputTabsContent">`;
            formats.forEach((format, index) => {
                const active = index === 0 ? 'show active' : '';
                const content = result.output_formats[format];
                
                html += `<div class="tab-pane fade ${active}" id="${format}" role="tabpanel">
                    <div class="math-output format-tab-content">`;
                
                if (format === 'html') {
                    html += content; // Render HTML directly
                } else {
                    html += `<pre><code>${this.escapeHtml(content)}</code></pre>`;
                }
                
                html += `</div></div>`;
            });
            html += `</div>`;
            html += `</div>`;
        }

        container.innerHTML = html;

        // Update domain analysis
        this.displayDomainAnalysis(result.domain_analysis);

        // Update performance metrics
        this.displayPerformanceMetrics(result);

        // Re-render MathJax if available
        if (window.MathJax) {
            MathJax.typesetPromise([container]).catch((err) => console.log(err.message));
        }
    }

    displayDomainAnalysis(domainAnalysis) {
        const container = document.getElementById('domain-analysis');
        
        if (!domainAnalysis || Object.keys(domainAnalysis).length === 0) {
            container.innerHTML = '<p class="text-muted">No domain-specific features detected.</p>';
            return;
        }

        let html = '';
        
        Object.entries(domainAnalysis).forEach(([domain, analysis]) => {
            if (analysis.detected) {
                html += `<div class="domain-badge">
                    <strong>${domain.replace('_', ' ').toUpperCase()}</strong>`;
                
                // Add domain-specific details
                if (domain === 'complex_numbers') {
                    if (analysis.magnitude !== null) {
                        html += `<br>Magnitude: ${analysis.magnitude.toFixed(3)}`;
                    }
                    if (analysis.argument !== null) {
                        html += `<br>Argument: ${analysis.argument.toFixed(1)}°`;
                    }
                } else if (domain === 'differential_equations') {
                    html += `<br>Type: ${analysis.type}`;
                    html += `<br>Order: ${analysis.order}`;
                    html += `<br>Linearity: ${analysis.linearity}`;
                } else if (domain === 'probability') {
                    html += `<br>Distribution: ${analysis.distribution_type}`;
                    if (analysis.properties && analysis.properties.mean !== undefined) {
                        html += `<br>Mean: ${analysis.properties.mean}`;
                    }
                } else if (domain === 'set_theory') {
                    if (analysis.cardinality !== null) {
                        html += `<br>Cardinality: ${analysis.cardinality}`;
                    }
                    html += `<br>Finite: ${analysis.is_finite}`;
                }
                
                html += `</div>`;
            }
        });

        container.innerHTML = html || '<p class="text-muted">No domain-specific features detected.</p>';
    }

    displayPerformanceMetrics(result) {
        const container = document.getElementById('performance-info');
        
        let html = '';
        
        if (result.processing_time !== undefined) {
            html += `<div class="performance-metric">
                <span>Processing Time:</span>
                <span class="performance-value">${(result.processing_time * 1000).toFixed(2)} ms</span>
            </div>`;
        }

        if (result.expression_id) {
            html += `<div class="performance-metric">
                <span>Expression ID:</span>
                <span class="performance-value">${result.expression_id}</span>
            </div>`;
        }

        if (result.output_formats) {
            html += `<div class="performance-metric">
                <span>Output Formats:</span>
                <span class="performance-value">${Object.keys(result.output_formats).length}</span>
            </div>`;
        }

        container.innerHTML = html;
    }

    displayError(errorMessage) {
        const container = document.getElementById('results-container');
        container.innerHTML = `<div class="error-message">
            <strong>Error:</strong> ${this.escapeHtml(errorMessage)}
        </div>`;
    }

    showLoading() {
        const container = document.getElementById('results-container');
        container.innerHTML = '<div class="text-center p-4"><div class="loading">Loading...</div></div>';
    }

    hideLoading() {
        // Loading is hidden when results are displayed
    }

    updateConnectionStatus(status, type) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status;
        statusElement.className = `badge bg-${type}`;
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, (m) => map[m]);
    }

    // Share functionality
    shareExpression() {
        if (!this.currentExpressionId) {
            this.displayError('No expression to share');
            return;
        }

        const shareUrl = `${window.location.origin}/share/${this.currentExpressionId}`;
        document.getElementById('share-link').value = shareUrl;
        
        const modal = new bootstrap.Modal(document.getElementById('shareModal'));
        modal.show();
    }

    copyShareLink() {
        const input = document.getElementById('share-link');
        input.select();
        input.setSelectionRange(0, 99999); // For mobile devices
        
        navigator.clipboard.writeText(input.value).then(() => {
            this.showToast('Link copied to clipboard!', 'success');
        }).catch(() => {
            // Fallback for older browsers
            document.execCommand('copy');
            this.showToast('Link copied to clipboard!', 'success');
        });
    }

    // Export functionality
    exportExpression() {
        if (!this.currentExpressionId) {
            this.displayError('No expression to export');
            return;
        }

        const modal = new bootstrap.Modal(document.getElementById('exportModal'));
        modal.show();
    }

    downloadFormat(format) {
        if (!this.currentExpressionId) {
            return;
        }

        const url = `/api/export/${this.currentExpressionId}/${format}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = `expression_${this.currentExpressionId}.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('exportModal'));
        modal.hide();

        this.showToast(`Exported as ${format.toUpperCase()}`, 'success');
    }

    showToast(message, type = 'info') {
        // Create toast container if it doesn't exist
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }

        // Create toast
        const toastId = 'toast_' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header">
                    <strong class="me-auto">MathML Parser</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${this.escapeHtml(message)}
                </div>
            </div>
        `;

        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        
        const toast = new bootstrap.Toast(document.getElementById(toastId));
        toast.show();

        // Remove toast element after it's hidden
        document.getElementById(toastId).addEventListener('hidden.bs.toast', () => {
            document.getElementById(toastId).remove();
        });
    }
}

// Global functions for HTML onclick handlers
function parseExpression() {
    window.mathmlInterface.parseExpression();
}

function clearInput() {
    document.getElementById('expression-input').value = '';
    document.getElementById('expression-input').style.height = 'auto';
    document.getElementById('results-container').innerHTML = '<p class="text-muted">Enter a mathematical expression to see the parsed results.</p>';
    document.getElementById('domain-analysis').innerHTML = '<p class="text-muted">Domain-specific analysis will appear here.</p>';
    document.getElementById('performance-info').innerHTML = '<p class="text-muted">Parse an expression to see performance metrics.</p>';
    
    // Disable buttons
    document.getElementById('share-btn').disabled = true;
    document.getElementById('export-btn').disabled = true;
    
    window.mathmlInterface.currentExpressionId = null;
}

function setExample(expression) {
    document.getElementById('expression-input').value = expression;
    parseExpression();
}

function shareExpression() {
    window.mathmlInterface.shareExpression();
}

function exportExpression() {
    window.mathmlInterface.exportExpression();
}

function copyShareLink() {
    window.mathmlInterface.copyShareLink();
}

function downloadFormat(format) {
    window.mathmlInterface.downloadFormat(format);
}

function showHistory() {
    // TODO: Implement history functionality
    window.mathmlInterface.showToast('History feature coming soon!', 'info');
}

function showHelp() {
    // TODO: Implement help functionality
    window.mathmlInterface.showToast('Help documentation coming soon!', 'info');
}

// Initialize interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mathmlInterface = new MathMLWebInterface();
});'''
    
    with open(os.path.join(static_dir, 'js', 'app.js'), 'w', encoding='utf-8') as f:
        f.write(js_content)


def main():
    """Main function to run the web interface."""
    print("🌐 Starting MathML Parser Web Interface...")
    
    # Create templates and static files
    create_templates()
    print("✅ Templates and static files created")
    
    # Configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Starting server on http://{host}:{port}")
    print("📱 Web interface features:")
    print("   • Real-time mathematical expression parsing")
    print("   • Multiple output format visualization")
    print("   • Domain-specific analysis")
    print("   • Expression sharing and export")
    print("   • Responsive design for mobile and desktop")
    print()
    print("💡 Example expressions to try:")
    print("   • Complex numbers: 3 + 4i, 2∠45°")
    print("   • Differential equations: dy/dx = x + y")
    print("   • Probability: X ~ N(0, 1), Y ~ Bin(10, 0.3)")
    print("   • Set theory: {1, 2, 3, 4}, A ∪ B")
    print()
    
    try:
        # Run the application
        socketio.run(
            app, 
            host=host, 
            port=port, 
            debug=debug_mode,
            allow_unsafe_werkzeug=True  # For development only
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())