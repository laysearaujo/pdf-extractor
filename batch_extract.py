# /seu-projeto-flask/batch_extract.py

import requests
import argparse
import os
import json
import time

API_URL = "http://127.0.0.1:8000/api/batch"

def load_schema(schema_path: str) -> dict:
    """Carrega o arquivo JSON do schema."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler o arquivo de schema: {e}")
        exit(1)

def find_pdfs(folder_path: str) -> list:
    """Encontra todos os arquivos .pdf em uma pasta."""
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.abspath(os.path.join(root, file)))
    return pdf_files

def main():
    parser = argparse.ArgumentParser(description="Script para processamento em lote via API.")
    parser.add_argument("--folder", required=True, help="Pasta contendo os arquivos PDF.")
    parser.add_argument("--schema", required=True, help="Caminho para o arquivo JSON do schema.")
    parser.add_argument("--label", help="Label a ser usado. (Default: nome da pasta)")
    
    args = parser.parse_args()

    # Define o label
    label = args.label if args.label else os.path.basename(os.path.normpath(args.folder))
    
    # Carrega o schema
    print(f"Carregando schema de '{args.schema}'...")
    schema = load_schema(args.schema)
    
    # Encontra os PDFs
    print(f"Procurando PDFs em '{args.folder}'...")
    pdf_paths = find_pdfs(args.folder)
    if not pdf_paths:
        print("Nenhum arquivo PDF encontrado.")
        return

    print(f"Encontrados {len(pdf_paths)} PDFs. Preparando requisição em lote...")
    
    # Monta o payload para a API
    batch_payload = []
    for pdf_path in pdf_paths:
        batch_payload.append({
            "label": label,
            "extraction_schema": schema,
            "pdf_path": pdf_path # O servidor Flask acessa o caminho do arquivo
        })
    
    # Envia a requisição
    try:
        print(f"Enviando para {API_URL}...")
        start_time = time.time()
        response = requests.post(API_URL, json=batch_payload)
        response.raise_for_status() # Lança erro se a resposta for 4xx ou 5xx
        
        end_time = time.time()
        
        # Exibe os resultados
        results_data = response.json()
        
        print("\n--- RESULTADO DO LOTE ---")
        for res in results_data.get('results', []):
            if res.get('error'):
                print(f"[ERRO] Index {res['index']}: {res['error']}")
            else:
                print(f"[SUCESSO] Index {res['index']} ({res['time_taken']:.3f}s): {res['data']}")
        
        print("\n--- SUMÁRIO ---")
        print(f"Tempo total (API): {results_data.get('total_time'):.3f}s")
        print(f"Tempo médio (API): {results_data.get('avg_time'):.3f}s")
        print(f"Tempo total (Cliente): {end_time - start_time:.3f}s")

    except requests.exceptions.ConnectionError:
        print(f"\n[ERRO] Não foi possível conectar a {API_URL}.")
        print("Certifique-se de que o servidor Flask (app.py) está rodando.")
    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
