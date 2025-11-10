# Deployment Guide - Fixing CORS and Proxy Errors

## Changes Made

### 1. Flask Application (app.py)
- Added Flask-CORS support with proper configuration
- Added OPTIONS method support to all API endpoints
- Added explicit CORS headers to all responses
- Improved error handling with proper HTTP status codes
- Added health check endpoint at `/health`
- Made host and port configurable via environment variables
- Added proper logging configuration

### 2. Requirements (requirements.txt)
- Added `flask-cors` dependency

### 3. Web Server Configuration (web.config)
- Added CORS headers at server level
- Added OPTIONS request handling
- Increased request size limits (50MB)
- Added timeout configuration
- Enabled detailed error reporting

### 4. JavaScript Client (main.js)
- Added proper error handling for fetch requests
- Added X-Requested-With headers
- Improved error messages with HTTP status codes

### 5. Environment Configuration
- Created `.env.production` for deployment settings

## Deployment Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   - Copy `.env.production` to `.env` and update values
   - Set `FLASK_ENV=production`
   - Set `FLASK_DEBUG=False`

3. **Test Locally**
   ```bash
   python app.py
   ```
   - Visit `http://localhost:5000/health` to verify

4. **Deploy to Server**
   - Ensure web.config is properly configured
   - Restart IIS/web server
   - Monitor logs for any remaining issues

## Common Issues Fixed

- **500 Internal Server Error**: Added proper error handling and logging
- **CORS Errors**: Added Flask-CORS and server-level CORS headers
- **OPTIONS Preflight**: Added OPTIONS method support
- **Proxy Timeouts**: Increased timeout limits
- **File Upload Issues**: Increased request size limits

## Monitoring

- Health check: `GET /health`
- Check logs in: `wfastcgi.log`
- Monitor error responses with proper HTTP status codes

## Testing

Test each endpoint:
- `GET /health` - Should return 200 OK
- `GET /api/start` - Should return clusters
- `POST /api/ask` - Test with audio file
- All endpoints should include CORS headers