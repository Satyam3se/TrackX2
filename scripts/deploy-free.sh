#!/bin/bash
# ============================================================================
# TrackX — free VPS deploy (Oracle Always-Free / any Ubuntu 22.04+ VM, $0)
# Run ON THE SERVER as a user with sudo rights:
#   git clone <your-repo-url> track-x3 && cd track-x3
#   chmod +x scripts/deploy-free.sh && ./scripts/deploy-free.sh
# Re-running is safe (idempotent): rebuilds, migrates, keeps DB volume.
# ============================================================================
set -euo pipefail

echo "==> 1/5 Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
  echo "    Installing Docker..."
  sudo apt-get update -y
  sudo apt-get install -y docker.io docker-compose-plugin git openssl ufw
  sudo systemctl enable --now docker
else
  echo "    Docker OK: $(docker --version)"
fi

echo "==> 2/5 Environment (.env)..."
if [ ! -f .env ]; then
  if [ -f .env.prod.example ]; then
    cp .env.prod.example .env
    echo "    Copied .env.prod.example -> .env"
  else
    echo "    ERROR: no .env and no .env.prod.example found." >&2
    exit 1
  fi
fi

# Generate secrets on first run if still placeholders.
if grep -q "change-me-generate" .env; then
  SECRET=$(openssl rand -hex 32)
  sed -i "s/^DJANGO_SECRET_KEY=.*/DJANGO_SECRET_KEY=${SECRET}/" .env
  echo "    Generated DJANGO_SECRET_KEY"
fi
if grep -q "change-me-db-password" .env; then
  DBPASS=$(openssl rand -hex 16)
  sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=${DBPASS}/" .env
  echo "    Generated POSTGRES_PASSWORD"
fi

# shellcheck disable=SC1091
set -a; source .env; set +a
if [ -z "${PUBLIC_HOST:-}" ] || [ "$PUBLIC_HOST" = "trackx-demo.duckdns.org" ]; then
  echo ""
  echo "    WARNING: PUBLIC_HOST is unset or still the example value."
  echo "    Get a free name at https://www.duckdns.org pointing at this VM's IP,"
  echo "    then edit .env (PUBLIC_HOST + ALLOWED_HOSTS) and re-run this script."
  echo "    Continuing with HTTP-only local boot so you can test via http://<server-ip>:3000 ..."
  COMPOSE_FILES="-f docker-compose.yml"
else
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
fi

echo "==> 3/5 Opening firewall (80/443/22)..."
sudo ufw allow 22/tcp  >/dev/null 2>&1 || true
sudo ufw allow 80/tcp  >/dev/null 2>&1 || true
sudo ufw allow 443/tcp >/dev/null 2>&1 || true
sudo ufw --force enable >/dev/null 2>&1 || true

echo "==> 4/5 Building + starting (first boot warms ANPR models, 1-3 min)..."
# shellcheck disable=SC2086
sudo docker compose $COMPOSE_FILES up --build -d
echo "    Waiting for backend..."
sleep 10
sudo docker compose $COMPOSE_FILES logs --tail=30 web || true

echo "==> 5/5 Status"
# shellcheck disable=SC2086
sudo docker compose $COMPOSE_FILES ps
echo ""
if [ "$COMPOSE_FILES" = "-f docker-compose.yml" ]; then
  echo "NEXT: test http://$(curl -s ifconfig.me 2>/dev/null || echo '<server-ip>'):3000"
  echo "THEN: set PUBLIC_HOST in .env to your DuckDNS name and re-run ./scripts/deploy-free.sh"
else
  echo "DONE: https://${PUBLIC_HOST} should be live in ~1-3 min (first boot downloads ANPR models)."
  echo "Create admin: sudo docker compose $COMPOSE_FILES exec web python manage.py createsuperuser"
fi
