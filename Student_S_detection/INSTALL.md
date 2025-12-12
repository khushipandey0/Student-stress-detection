# Installation Instructions

## Quick Setup

Run the setup script (requires sudo password):
```bash
./setup.sh
```

## Manual Setup

If you prefer to set up manually:

### Step 1: Install System Packages
```bash
sudo apt update
sudo apt install python3-pip python3-venv
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 3: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 4: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Using the Project

After setup, always activate the virtual environment first:
```bash
source venv/bin/activate
```

Then you can:
- Run the pipeline: `python run_pipeline.py`
- Start web app: `cd web_app && python app.py`

## Troubleshooting

### If pip is not found:
- Install python3-pip: `sudo apt install python3-pip`
- Or use: `python3 -m pip` instead of `pip`

### If venv creation fails:
- Install python3-venv: `sudo apt install python3-venv`

### If you see PEP 668 errors:
- Use a virtual environment (recommended)
- Or use: `pip install --break-system-packages` (not recommended)

