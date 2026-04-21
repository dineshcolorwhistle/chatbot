# Production Deployment Guide (Hostinger VPS)

Since you have already migrated the code to your Hostinger VPS via Git, the next step is to configure the server, install dependencies, build the project, and set up a process manager and web server (Nginx).

Here is the step-by-step blueprint to bring your chatbot online.

## Step 1: Install Server Dependencies

SSH into your Hostinger VPS and install the required packages (Nginx, Node.js, Python, PM2).

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install Python & Nginx
sudo apt install -y python3-pip python3-venv nginx

# Install Node.js (via NVM or directly)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install PM2 (Process Manager for keeping backend alive)
sudo npm install pm2 -g
```

## Step 2: Set Up the Backend (FastAPI)

Navigate to the backend directory, install Python libraries, and test it.

```bash
cd /path/to/your/chatbot/backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
nano .env # Add your production keys (OpenAI, Pinecone, SMTP, etc.)
```

### Start Backend with PM2

Instead of keeping the terminal open, use PM2 to run FastAPI in the background. Note: you must specify the path to your virtual environment's uvicorn binary.

```bash
# Still in the backend folder:
pm2 start "venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000" --name "chatbot-backend"

# Save the PM2 process so it restarts on server reboot
pm2 save
pm2 startup
```

## Step 3: Set Up the Frontend (React Widget)

You are deploying an embeddable widget or full app. Let's build the production assets.

```bash
cd /path/to/your/chatbot/frontend

# Install dependencies
npm install

# Build the Frontend 
# For Widget:
npm run build:widget

# For Full-page app (optional):
npm run build
```

The compiled assets are now in `frontend/dist-widget/` (or `frontend/dist/`).

## Step 4: Configure Nginx (Reverse Proxy)

Nginx will serve your frontend static files and route `/api` requests to your FastAPI backend securely.

1. Create a new Nginx config file:
```bash
sudo nano /etc/nginx/sites-available/chatbot
```

2. Paste the following configuration (replace `yourdomain.com` and `/path/to/your/chatbot` with your actual domain and paths). If you are only deploying the widget file, point the root to `dist-widget`.

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com; # Or your VPS IP

    # Serve the frontend files (Widget or App)
    root /path/to/your/chatbot/frontend/dist-widget; # Change to 'dist' if hosting full app
    index index.html widget.js;

    # Frontend Routing
    location / {
        try_files $uri $uri/ =404;
        
        # Adding CORS headers for widget access from any domain
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
    }

    # Backend API Routing
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

3. Enable the config and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration for syntax errors
sudo systemctl restart nginx
```

## Step 5: Secure with SSL (HTTPS)

It is critical to have an SSL certificate, especially if you are embedding the widget on other HTTPS domains.

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Generate and apply SSL certificate automatically
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

## Step 6: Verify Deployment

1. Visit `https://yourdomain.com/widget.js` – You should see your bundled JS file.
2. Visit `https://yourdomain.com/api/health` – You should get a JSON health check response from FastAPI.

You can then embed the widget on external client websites using your live API URL:
```html
<script
  src="https://yourdomain.com/widget.js"
  data-api-url="https://yourdomain.com"
  data-position="bottom-right"
></script>
```

## Troubleshooting
- **Backend logs:** `pm2 logs chatbot-backend`
- **Nginx errors:** `sudo tail -f /var/log/nginx/error.log`
- If you change any `.env` variable, restart the backend: `pm2 restart chatbot-backend`
