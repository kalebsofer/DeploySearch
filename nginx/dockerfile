FROM nginx:alpine

# Install certbot and dependencies
RUN apk add --no-cache \
    certbot \
    certbot-nginx \
    openssl \
    iputils \
    bind-tools \
    netcat-openbsd \
    dos2unix

# Remove default config and create required directories
RUN rm -f /etc/nginx/conf.d/default.conf
RUN mkdir -p /var/www/html
RUN mkdir -p /etc/letsencrypt
RUN mkdir -p /etc/nginx/conf.d

# Copy nginx config and SSL setup script
COPY conf.d/* /etc/nginx/conf.d/
COPY setup-ssl.sh /setup-ssl.sh

# Make script executable and ensure proper line endings
RUN chmod +x /setup-ssl.sh && \
    dos2unix /setup-ssl.sh 2>/dev/null || true && \
    ls -la /setup-ssl.sh  # Verify permissions

EXPOSE 80 443

# Copy the initial config to a backup location
RUN cp /etc/nginx/conf.d/default.conf /etc/nginx/conf.d/default.conf.orig

ENTRYPOINT ["/setup-ssl.sh"]