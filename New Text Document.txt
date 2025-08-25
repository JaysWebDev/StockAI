# GitHub Project Transfer Script - Customized for JaysWebDev/StockAI
# Prerequisites: Install GitHub CLI or create Personal Access Token

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
$gitignoreContent = @"
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
"@

if (-not (Test-Path ".gitignore")) {
    Write-Info "Creating .gitignore file..."
    $gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Success ".gitignore created"
}

# Create README.md if it doesn't exist
$readmeContent = @"
# StockAI - Advanced Portfolio Management System

Comprehensive stock analysis and portfolio management application with AI-driven insights and automated decision-making capabilities.

## 🚀 Features

- **Advanced Portfolio Analysis**: Multi-factor optimization with risk management
- **Stock Intelligence Engine**: AI-powered stock screening and analysis
- **Real-time Data Integration**: Multiple financial data API connections
- **Interactive Dashboard**: Web-based portfolio management interface
- **Automated Scheduling**: Background data updates and analysis pipelines
- **Risk Management**: Built-in VaR calculations and risk monitoring
- **Decision Engine**: Automated investment decision recommendations

## 📁 Project Structure

\`\`\`
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
├── data/                      # Data storage and databases
├── logs/                      # Application logs
├── tests/                     # Test suites
└── docs/                      # Documentation
\`\`\`

## 🛠️ Setup Instructions

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/JaysWebDev/StockAI.git
   cd StockAI
   \`\`\`

2. **Install Python dependencies:**
   \`\`\`bash
   pip install -r backend/requirements.txt
   \`\`\`

3. **Configure environment variables:**
   \`\`\`bash
   cp .env.example .env
   # Edit .env with your API keys
   \`\`\`

4. **Initialize database:**
   \`\`\`bash
   python backend/unified_stock_intelligence.py --init-db
   \`\`\`

5. **Run the application:**
   - **Backend**: \`python backend/unified_stock_intelligence.py\`
   - **Frontend**: Open \`frontend/index.html\` in browser
   - **Scheduler**: \`scheduler/run_daily_update.bat\`

## 🔧 Configuration

Create a \`.env\` file in the root directory:
\`\`\`env
# Financial Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINANCIAL_MODELING_PREP_API_KEY=your_fmp_key
FINNHUB_API_KEY=your_finnhub_key

# Database Configuration
DATABASE_URL=sqlite:///data/stocks.db

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
PORTFOLIO_SIZE=10000
RISK_FREE_RATE=0.05

# Analysis Parameters
LOOKBACK_PERIOD=252
REBALANCE_FREQUENCY=30
MAX_POSITION_SIZE=0.1
\`\`\`

## 📊 Usage

### Portfolio Management
1. Open the web interface at \`frontend/index.html\`
2. Input your portfolio parameters and constraints
3. Run optimization and analysis tools
4. Review recommendations and risk metrics

### Automated Analysis
- **Daily Updates**: Run \`scheduler/run_daily_update.bat\` for daily data refresh
- **Continuous Monitoring**: Set up scheduled tasks for real-time monitoring
- **Custom Analysis**: Use the unified intelligence engine for ad-hoc analysis

### API Integration
The system integrates with multiple data sources:
- **Alpha Vantage**: Real-time and historical stock data
- **Financial Modeling Prep**: Fundamental analysis data
- **Finnhub**: News sentiment and earnings data
- **SEC EDGAR**: Institutional holdings and filings

## 🔒 Security

- All API keys stored in environment variables
- Database connections secured with connection pooling
- Input validation and sanitization implemented
- Regular security updates for dependencies

## 📈 Live Demo

**GitHub Pages**: https://JaysWebDev.github.io/StockAI/

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: \`git checkout -b feature/new-feature\`
3. Commit changes: \`git commit -am 'Add new feature'\`
4. Push to branch: \`git push origin feature/new-feature\`
5. Submit pull request

## 📝 License

This project is for educational and personal use.

---

**Developer**: JaysWebDev | **Repository**: https://github.com/JaysWebDev/StockAI
"@

if (-not (Test-Path "README.md")) {
    Write-Info "Creating README.md..."
    $readmeContent | Out-File -FilePath "README.md" -Encoding UTF8
    Write-Success "README.md created"
}

# Stage all files
Write-Info "Staging files for commit..."
git add .

# Initial commit
Write-Info "Creating initial commit..."
git commit -m "Initial commit: Portfolio management application

- Backend Python modules for stock analysis
- Frontend HTML/JS dashboard
- Scheduler for automated updates
- Configuration and documentation files"

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