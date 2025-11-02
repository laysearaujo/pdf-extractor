#!/usr/bin/env python3
"""
Script CLI para processamento em lote de PDFs
Uso: python batch_extract.py input.json output.json
"""

import json
import sys
import os
from pathlib import Path
import time
from extractor import SmartExtractor


def load_batch_file(filepath: str) -> list:
    """Carrega arquivo JSON com requisições em lote"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Arquivo deve conter um array JSON")
    
    return data


def process_batch(extractor: SmartExtractor, requests: list, base_path: str = "") -> list:
    """Processa lista de requisições em série"""
    results = []
    total_time = 0
    total_cost = 0
    
    print(f"\n{'='*70}")
    print(f"Processando {len(requests)} documentos em série...")
    print(f"{'='*70}\n")
    
    for i, req in enumerate(requests, 1):
        print(f"[{i}/{len(requests)}] Processando: {req.get('pdf_path', 'unknown')}")
        
        start_time = time.time()
        
        try:
            label = req.get('label')
            extraction_schema = req.get('extraction_schema')
            pdf_path = req.get('pdf_path')
            
            # Valida campos obrigatórios
            if not all([label, extraction_schema, pdf_path]):
                raise ValueError("Campos obrigatórios: label, extraction_schema, pdf_path")
            
            # Resolve caminho do PDF
            if base_path:
                pdf_path = os.path.join(base_path, pdf_path)
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
            
            # Extrai dados
            data = extractor.extract(label, extraction_schema, pdf_path)
            
            time_taken = time.time() - start_time
            total_time += time_taken
            
            result = {
                'index': i - 1,
                'label': label,
                'pdf_path': req.get('pdf_path'),
                'success': True,
                'data': data,
                'time_taken': round(time_taken, 3)
            }
            
            print(f"  ✓ Sucesso ({time_taken:.2f}s)")
            print(f"  Campos extraídos: {list(data.keys())}")
            
        except Exception as e:
            result = {
                'index': i - 1,
                'label': req.get('label'),
                'pdf_path': req.get('pdf_path'),
                'success': False,
                'error': str(e),
                'time_taken': round(time.time() - start_time, 3)
            }
            print(f"  ✗ Erro: {str(e)}")
        
        results.append(result)
        print()
    
    # Estatísticas finais
    success_count = sum(1 for r in results if r.get('success'))
    avg_time = total_time / len(requests) if requests else 0
    
    print(f"{'='*70}")
    print(f"ESTATÍSTICAS FINAIS")
    print(f"{'='*70}")
    print(f"Total de documentos: {len(requests)}")
    print(f"Sucessos: {success_count}")
    print(f"Falhas: {len(requests) - success_count}")
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Tempo médio: {avg_time:.2f}s")
    print(f"{'='*70}\n")
    
    return results


def save_results(results: list, output_path: str):
    """Salva resultados em arquivo JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Resultados salvos em: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Uso: python batch_extract.py <input.json> <output.json> [base_path]")
        print("\nExemplo:")
        print("  python batch_extract.py requests.json results.json ./data")
        print("\nFormato do input.json:")
        print("""
[
  {
    "label": "carteira_oab",
    "extraction_schema": {
      "nome": "Nome do profissional",
      "inscricao": "Número de inscrição"
    },
    "pdf_path": "oab_1.pdf"
  }
]
        """)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    base_path = sys.argv[3] if len(sys.argv) > 3 else ""
    
    # Valida API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Erro: OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        print("\nDefina a variável de ambiente:")
        print("  export OPENAI_API_KEY='sua-chave-aqui'")
        sys.exit(1)
    
    try:
        # Carrega requisições
        print(f"Carregando requisições de: {input_file}")
        requests = load_batch_file(input_file)
        print(f"✓ {len(requests)} requisições carregadas")
        
        # Inicializa extrator
        print("Inicializando extrator...")
        extractor = SmartExtractor(api_key)
        print("✓ Extrator pronto")
        
        # Processa em lote
        results = process_batch(extractor, requests, base_path)
        
        # Salva resultados
        save_results(results, output_file)
        
        # Verifica se houve falhas
        failures = [r for r in results if not r.get('success')]
        if failures:
            print(f"\n⚠️  {len(failures)} documento(s) falharam:")
            for fail in failures:
                print(f"  - {fail['pdf_path']}: {fail['error']}")
            sys.exit(1)
        else:
            print("\n✅ Todos os documentos foram processados com sucesso!")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Erro fatal: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()