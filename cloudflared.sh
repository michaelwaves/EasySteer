#!/bin/bash

set -e

### -----------------------------------------
### CONFIG — EDIT THESE TWO VALUES
### -----------------------------------------

TUNNEL_NAME="myapp"
DOMAIN_1="confess.it.com"      # domain 1
DOMAIN_2="aimoodring.com"      # domain 2
LOCAL_PORT="8080"              # your app port

### -----------------------------------------
### INSTALL CLOUDFLARED
### -----------------------------------------

echo "➡ Installing cloudflared..."
sudo mkdir -p /usr/local/bin
sudo wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
sudo chmod +x /usr/local/bin/cloudflared

echo "✔ cloudflared installed: $(cloudflared --version)"

### -----------------------------------------
### CLOUDLFARE LOGIN (REQUIRES BROWSER)
### -----------------------------------------

echo ""
echo "🚨 IMPORTANT: Cloudflare login required"
echo "A URL will appear. Open it in your browser to authenticate."
echo "Press enter to continue..."
read

cloudflared tunnel login

### -----------------------------------------
### CREATE TUNNEL
### -----------------------------------------

echo "➡ Creating tunnel: $TUNNEL_NAME..."
TUNNEL_ID=$(cloudflared tunnel create "$TUNNEL_NAME" | grep -oE "[0-9a-fA-F-]{36}")

echo "✔ Tunnel created!"
echo "Tunnel ID: $TUNNEL_ID"

CREDS_FILE="/root/.cloudflared/${TUNNEL_ID}.json"

### -----------------------------------------
### WRITE /etc/cloudflared/config.yml
### -----------------------------------------

echo "➡ Writing /etc/cloudflared/config.yml..."
sudo mkdir -p /etc/cloudflared

sudo tee /etc/cloudflared/config.yml > /dev/null <<EOF
tunnel: $TUNNEL_ID
credentials-file: $CREDS_FILE

ingress:
  - hostname: $DOMAIN_1
    service: http://localhost:$LOCAL_PORT
  - hostname: $DOMAIN_2
    service: http://localhost:$LOCAL_PORT
  - service: http_status:404
EOF

echo "✔ Config created at /etc/cloudflared/config.yml"

### -----------------------------------------
### CREATE DNS RECORDS
### -----------------------------------------

echo "➡ Creating DNS routes..."

cloudflared tunnel route dns "$TUNNEL_NAME" "$DOMAIN_1"
cloudflared tunnel route dns "$TUNNEL_NAME" "$DOMAIN_2"

echo "✔ DNS routes created"

### -----------------------------------------
### RUN TUNNEL
### -----------------------------------------

echo ""
echo "➡ Starting Cloudflare Tunnel..."
echo "You can stop it with CTRL+C"
echo ""

cloudflared tunnel run "$TUNNEL_NAME"
