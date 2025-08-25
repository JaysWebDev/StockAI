# Clean GitHub Transfer Script for StockAI
param(
    [Parameter(Mandatory=$false)]
    [string]$GitHubUsername = "JaysWebDev",
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "StockAI",
    
    [Parameter(Mandatory=$false)]
    [string]$LocalProjectPath = "D:\StockAI_Master",
    
    [Parameter(Mandatory=$false)]
    [string]$PersonalAccessToken = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$UseGitHubCLI = $false
)

# Color output functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️ $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️ $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }

Write-Info "Starting GitHub project transfer for StockAI..."
Write-Info "Project Path: $LocalProjectPath"
Write-Info "Repository: https://github.com/$GitHubUsername/$RepositoryName"

# Validate project path exists
if (-not (Test-Path $LocalProjectPath)) {
    Write-Error "Project path does not exist: $LocalProjectPath"
    exit 1
}

# Change to project directory
Set-Location $LocalProjectPath

# Initialize git if not already initialized
if (-not (Test-Path ".git")) {
    Write-Info "Initializing Git repository..."
    git init
    Write-Success "Git repository initialized"
} else {
    Write-Info "Git repository already exists"
}

# Create .gitignore if it doesn't exist
$gitignoreContent = @'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv/

# Environment variables
.env
.env.local
.env.production
config.ini

# Database
*.db
*.sqlite
*.sqlite3

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# API Keys and Secrets
*api_key*
*secret*
*password*
credentials.json
'@

if (-not (Test-Path ".gitignore")) {
    Write-Info "Creating .gitignore file..."
    $gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Success ".gitignore created"
}

# Create README.md if it doesn't exist
$readmeContent = @'
# StockAI - Advanced Portfolio Management System

Comprehensive stock analysis and portfolio management application with AI-driven insights.

## Features
- Advanced Portfolio Analysis with Multi-factor optimization
- Stock Intelligence Engine with AI-powered screening
- Real-time Data Integration from multiple APIs
- Interactive Dashboard for portfolio management
- Automated Scheduling for data updates
- Risk Management with VaR calculations
- Decision Engine for investment recommendations

## Project Structure
```
StockAI_Master/
├── backend/                    # Python backend modules
│   ├── unified_stock_intelligence.py
│   ├── requirements.txt
│   └── db.py
├── frontend/                   # HTML/JS frontend
│   ├── index.html
│   └── methodology.html
├── scheduler/                  # Automated tasks
│   ├── daily_scheduler.py
│   └── run_daily_update.bat
├── data/                      # Data storage
├── logs/                      # Application logs
└── tests/                     # Test suites
```

## Setup Instructions
1. Clone: `git clone https://github.com/JaysWebDev/StockAI.git`
2. Install: `pip install -r backend/requirements.txt`
3. Configure: Copy `.env.example` to `.env` and add your API keys
4. Run: `python backend/unified_stock_intelligence.py`

## Configuration
Create `.env` file with:
```
ALPHA_VANTAGE_API_KEY=your_key_here
FINANCIAL_MODELING_PREP_API_KEY=your_key_here
DATABASE_URL=sqlite:///data/stocks.db
```

## Live Demo
https://JaysWebDev.github.io/StockAI/

## Developer
JaysWebDev | Repository: https://github.com/JaysWebDev/StockAI
'@

if (-not (Test-Path "README.md")) {
    Write-Info "Creating README.md..."
    $readmeContent | Out-File -FilePath "README.md" -Encoding UTF8
    Write-Success "README.md created"
}

# Create .env.example template
$envTemplate = @'
# API Configuration
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINANCIAL_MODELING_PREP_API_KEY=your_fmp_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# Database Configuration
DATABASE_URL=sqlite:///data/stocks.db

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
DEFAULT_PORTFOLIO_SIZE=10000
RISK_FREE_RATE=0.05
'@

if (-not (Test-Path ".env.example")) {
    Write-Info "Creating .env.example template..."
    $envTemplate | Out-File -FilePath ".env.example" -Encoding UTF8
    Write-Success ".env.example created"
}

# Stage all files
Write-Info "Staging files for commit..."
git add .

# Initial commit
Write-Info "Creating initial commit..."
git commit -m "Initial commit: StockAI Portfolio Management System

Features:
- Backend Python modules for stock analysis and portfolio management
- Frontend HTML/JavaScript dashboard with interactive features
- Scheduler for automated data updates and analysis
- Comprehensive configuration and documentation
- Ready for GitHub Pages deployment

Components:
- unified_stock_intelligence.py: Core analysis engine
- Interactive web interface for portfolio management
- Automated scheduling and data pipeline
- Risk management and optimization tools"

# Create remote repository and push
if ($UseGitHubCLI) {
    Write-Info "Using GitHub CLI to connect to existing repository..."
    
    # Check if GitHub CLI is installed
    if (-not (Get-Command "gh" -ErrorAction SilentlyContinue)) {
        Write-Error "GitHub CLI not found. Please install it or use Personal Access Token method."
        Write-Info "Install from: https://cli.github.com/"
        exit 1
    }
    
    # Add remote and push to existing repository
    Write-Info "Connecting to existing GitHub repository..."
    git remote add origin "https://github.com/$GitHubUsername/$RepositoryName.git"
    git branch -M main
    git push -u origin main --force
    Write-Success "Repository updated using GitHub CLI"
    
} else {
    # Using Personal Access Token method
    if ($PersonalAccessToken -eq "") {
        Write-Error "Personal Access Token required when not using GitHub CLI"
        Write-Info "Create token at: https://github.com/settings/tokens"
        Write-Info "Required scopes: repo, workflow"
        exit 1
    }
    
    # Set up remote with token authentication
    $remoteUrl = "https://$PersonalAccessToken@github.com/$GitHubUsername/$RepositoryName.git"
    
    Write-Info "Adding remote origin to existing repository..."
    git remote add origin $remoteUrl
    
    Write-Info "Pushing to existing GitHub repository..."
    git branch -M main
    git push -u origin main --force
    
    Write-Success "Repository updated on GitHub"
}

# Enable GitHub Pages (if using GitHub CLI)
if ($UseGitHubCLI) {
    Write-Info "Enabling GitHub Pages..."
    try {
        gh api -X POST /repos/$GitHubUsername/$RepositoryName/pages -f source='{"branch":"main","path":"/"}'
        Write-Success "GitHub Pages enabled"
        Write-Info "Your site will be available at: https://$GitHubUsername.github.io/$RepositoryName/"
    } catch {
        Write-Warning "Could not automatically enable GitHub Pages. Enable manually in repository settings."
    }
}

Write-Success "StockAI project successfully transferred to GitHub!"
Write-Info "Repository URL: https://github.com/JaysWebDev/StockAI"

if (-not $UseGitHubCLI) {
    Write-Info "To enable GitHub Pages:"
    Write-Info "1. Go to https://github.com/JaysWebDev/StockAI/settings/pages"
    Write-Info "2. Select 'Deploy from a branch' and choose 'main' branch"
    Write-Info "3. Your site will be at: https://JaysWebDev.github.io/StockAI/"
}

Write-Info "Next steps:"
Write-Info "- Update any hardcoded API keys to use environment variables"
Write-Info "- Configure GitHub Actions for automated deployment if needed"
Write-Info "- Update DNS settings to point to GitHub Pages"