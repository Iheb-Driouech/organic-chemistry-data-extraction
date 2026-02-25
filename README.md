# Data extraction from organic chemistry literature

Pipeline to:
- extract structured reaction data (reactants/products/conditions) from JSON procedures using OpenAI
- structure to table
- standardize reaction times
- generate SMILES with traceability (PubChem / OPSIN / LLM)

## Setup (Windows)
1) Create venv:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install deps:
   pip install -r .\src\requirements.txt

3) Set API key (PowerShell):
   setx OPENAI_API_KEY "YOUR_KEY_HERE"
   # Close and reopen terminal

4) Run:
   python .\src\pipeline.py --input .\data\input_test.json --out .\outputs
