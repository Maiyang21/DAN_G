"""
Simple Authentication Service for DAN_G Platform
Basic session-based authentication without JWT complexity
"""

import logging
from functools import wraps
from flask import request, jsonify, session

logger = logging.getLogger(__name__)

class SimpleAuth:
    """Simple authentication service"""
    
    def __init__(self):
        self.demo_users = {
            'admin': {
                'id': '1',
                'username': 'admin',
                'email': 'admin@dan-g-platform.com',
                'password': 'admin123',  # In production, use hashed passwords
                'name': 'Admin User'
            }
        }
    
    def authenticate_user(self, username: str, password: str) -> dict:
        """Authenticate user with username and password"""
        try:
            if username in self.demo_users:
                user = self.demo_users[username]
                if user['password'] == password:
                    # Remove password from user data
                    user_data = {k: v for k, v in user.items() if k != 'password'}
                    return {
                        'success': True,
                        'user': user_data
                    }
            
            return {
                'success': False,
                'error': 'Invalid credentials'
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return {
                'success': False,
                'error': 'Authentication failed'
            }
    
    def create_session(self, user_data: dict) -> bool:
        """Create user session"""
        try:
            session['user_id'] = user_data['id']
            session['username'] = user_data['username']
            session['email'] = user_data['email']
            session['authenticated'] = True
            return True
        except Exception as e:
            logger.error(f"Session creation error: {str(e)}")
            return False
    
    def get_current_user(self) -> dict:
        """Get current user from session"""
        if session.get('authenticated'):
            return {
                'id': session.get('user_id'),
                'username': session.get('username'),
                'email': session.get('email')
            }
        return None
    
    def logout_user(self) -> bool:
        """Logout user and clear session"""
        try:
            session.clear()
            return True
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return session.get('authenticated', False)

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = SimpleAuth()
        if not auth.is_authenticated():
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_auth_optional(f):
    """Decorator for optional authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow the function to run regardless of auth status
        return f(*args, **kwargs)
    return decorated_function


