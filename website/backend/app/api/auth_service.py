"""
Authentication Service for DAN_G Platform
Handles user authentication, authorization, and session management
"""

import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from functools import wraps
from flask import request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

class AuthService:
    """Authentication and authorization service"""
    
    def __init__(self):
        self.secret_key = current_app.config.get('SECRET_KEY', 'dev-secret-key')
        self.jwt_expiration = current_app.config.get('JWT_EXPIRATION', 3600)  # 1 hour
        self.refresh_expiration = current_app.config.get('REFRESH_EXPIRATION', 604800)  # 7 days
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            return generate_password_hash(password)
        except Exception as e:
            logger.error(f"Error hashing password: {str(e)}")
            raise
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return check_password_hash(hashed, password)
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False
    
    def generate_tokens(self, user_id: int, username: str, role: str) -> Dict[str, str]:
        """Generate JWT access and refresh tokens"""
        try:
            # Access token payload
            access_payload = {
                'user_id': user_id,
                'username': username,
                'role': role,
                'type': 'access',
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=self.jwt_expiration)
            }
            
            # Refresh token payload
            refresh_payload = {
                'user_id': user_id,
                'username': username,
                'type': 'refresh',
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=self.refresh_expiration)
            }
            
            # Generate tokens
            access_token = jwt.encode(access_payload, self.secret_key, algorithm='HS256')
            refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm='HS256')
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expires_in': self.jwt_expiration
            }
            
        except Exception as e:
            logger.error(f"Error generating tokens: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access token using refresh token"""
        try:
            payload = self.verify_token(refresh_token)
            
            if not payload or payload.get('type') != 'refresh':
                return None
            
            # Generate new access token
            access_payload = {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'role': payload.get('role', 'user'),
                'type': 'access',
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=self.jwt_expiration)
            }
            
            access_token = jwt.encode(access_payload, self.secret_key, algorithm='HS256')
            
            return {
                'access_token': access_token,
                'expires_in': self.jwt_expiration
            }
            
        except Exception as e:
            logger.error(f"Error refreshing token: {str(e)}")
            return None
    
    def require_auth(self, f):
        """Decorator to require authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None
            
            # Get token from Authorization header
            auth_header = request.headers.get('Authorization')
            if auth_header:
                try:
                    token = auth_header.split(' ')[1]  # Bearer <token>
                except IndexError:
                    return jsonify({'error': 'Invalid authorization header format'}), 401
            
            if not token:
                return jsonify({'error': 'Authorization token required'}), 401
            
            # Verify token
            payload = self.verify_token(token)
            if not payload:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Add user info to request context
            request.current_user = {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'role': payload.get('role', 'user')
            }
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def require_role(self, required_roles):
        """Decorator to require specific roles"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not hasattr(request, 'current_user'):
                    return jsonify({'error': 'Authentication required'}), 401
                
                user_role = request.current_user.get('role', 'user')
                if user_role not in required_roles:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'strength': self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> str:
        """Calculate password strength"""
        score = 0
        
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if any(c.isupper() for c in password):
            score += 1
        if any(c.islower() for c in password):
            score += 1
        if any(c.isdigit() for c in password):
            score += 1
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        
        if score <= 2:
            return "weak"
        elif score <= 4:
            return "medium"
        else:
            return "strong"
    
    def generate_password_reset_token(self, user_id: int, email: str) -> str:
        """Generate password reset token"""
        try:
            payload = {
                'user_id': user_id,
                'email': email,
                'type': 'password_reset',
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=1)  # 1 hour expiration
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            return token
            
        except Exception as e:
            logger.error(f"Error generating password reset token: {str(e)}")
            raise
    
    def verify_password_reset_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify password reset token"""
        try:
            payload = self.verify_token(token)
            
            if not payload or payload.get('type') != 'password_reset':
                return None
            
            return payload
            
        except Exception as e:
            logger.error(f"Error verifying password reset token: {str(e)}")
            return None
    
    def log_authentication_attempt(self, username: str, success: bool, ip_address: str = None):
        """Log authentication attempt"""
        try:
            logger.info(f"Authentication attempt - Username: {username}, Success: {success}, IP: {ip_address}")
            
            # In a production system, you might want to store this in a database
            # for security monitoring and audit purposes
            
        except Exception as e:
            logger.error(f"Error logging authentication attempt: {str(e)}")
    
    def check_rate_limit(self, ip_address: str, username: str = None) -> bool:
        """Check if user/IP is rate limited"""
        try:
            # In a production system, implement rate limiting using Redis
            # For now, return True (not rate limited)
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            return True
    
    def get_user_permissions(self, user_id: int, role: str) -> Dict[str, bool]:
        """Get user permissions based on role"""
        permissions = {
            'can_upload_data': True,
            'can_generate_forecast': True,
            'can_view_history': True,
            'can_delete_data': False,
            'can_manage_users': False,
            'can_view_metrics': False,
            'can_manage_system': False
        }
        
        if role == 'admin':
            permissions.update({
                'can_delete_data': True,
                'can_manage_users': True,
                'can_view_metrics': True,
                'can_manage_system': True
            })
        elif role == 'analyst':
            permissions.update({
                'can_view_metrics': True
            })
        
        return permissions


