#!/bin/sh

# Wait for DNS to be ready and services to be available
wait_for_service() {
    local service=$1
    local port=$2
    echo "Waiting for $service to be ready..."
    
    # Wait for DNS resolution
    until nslookup $service >/dev/null 2>&1; do
        echo "Waiting for $service DNS..."
        sleep 3
    done
    
    # Wait for ping
    until ping -c1 $service >/dev/null 2>&1; do
        echo "Waiting for $service ping..."
        sleep 3
    done
    
    # Wait for port
    until nc -z $service $port >/dev/null 2>&1; do
        echo "Waiting for $service port $port..."
        sleep 3
    done
    
    echo "$service is ready!"
}

# Ensure the config file exists
if [ ! -f "/etc/nginx/conf.d/default.conf" ]; then
    echo "Restoring original config..."
    cp /etc/nginx/conf.d/default.conf.orig /etc/nginx/conf.d/default.conf
fi

# Wait for both services with their ports
wait_for_service frontend 8501
wait_for_service backend 8051

# Test direct HTTP connection to services
echo "Testing frontend HTTP..."
curl -v http://frontend:8501/_stcore/health || exit 1
echo "Testing backend HTTP..."
curl -v http://backend:8051/health || exit 1

# Request SSL certificate with non-interactive renewal
certbot certonly --standalone \
    --email kaleb@softmaxco.io \
    --agree-tos \
    --no-eff-email \
    --non-interactive \
    --keep-until-expiring \
    -d simplesearchengine.com \
    -d www.simplesearchengine.com

# Configure nginx to use the SSL certificate
if [ -d "/etc/letsencrypt/live/simplesearchengine.com" ]; then
    echo "Creating SSL configuration..."
    cat > /etc/nginx/conf.d/default.conf <<EOF
# Add resolver for Docker DNS with longer timeout and retries
resolver 127.0.0.11 ipv6=off valid=30s;

upstream frontend_upstream {
    server frontend:8501 max_fails=3 fail_timeout=30s;
}

upstream backend_upstream {
    server backend:8051 max_fails=3 fail_timeout=30s;
}

# HTTP server (redirect to HTTPS)
server {
    listen 80;
    server_name simplesearchengine.com www.simplesearchengine.com;
    
    # Allow certbot authentication
    location /.well-known/acme-challenge/ {
        root /var/www/html;
        try_files \$uri =404;
    }

    # Redirect all other HTTP traffic to HTTPS
    location / {
        return 301 https://\$host\$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl;
    server_name simplesearchengine.com www.simplesearchengine.com;
    
    ssl_certificate /etc/letsencrypt/live/simplesearchengine.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/simplesearchengine.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    
    location / {
        proxy_pass http://frontend_upstream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /api/ {
        proxy_pass http://backend_upstream/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
fi

# Test nginx configuration
echo "Testing nginx configuration..."
nginx -t || exit 1

# Start nginx
echo "Starting nginx..."
nginx -g 'daemon off;'