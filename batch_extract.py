import json
import os
import argparse
import logging
import time
from dotenv import load_dotenv
from extractor import SmartExtractor, ExtractRequest

# --- Configuração de Logging ---
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('batch_script')
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
log.addHandler(handler)
log.propagate = False

# --- CONSTANTES ---
OUTPUT_FILENAME = "extraction_results.json"
KB_FILENAME = "knowledge_base.json"


def load_kb(extractor_instance: SmartExtractor, kb_path: str):
    """Carrega o Knowledge Base, se existir."""
    if os.path.exists(kb_path):
        try:
            extractor_instance.import_kb(kb_path)
            log.info(f"Knowledge base carregado de {kb_path}")
        except Exception as e:
            log.error(f"Falha ao carregar KB de {kb_path}: {e}")
    else:
        log.warning(f"Nenhum KB encontrado em {kb_path}. Iniciando um novo.")

def save_kb(extractor_instance: SmartExtractor, kb_path: str):
    """Salva o Knowledge Base."""
    try:
        extractor_instance.export_kb(kb_path)
        log.info(f"Knowledge base salvo em {kb_path}")
    except Exception as e:
        log.error(f"Falha ao salvar KB em {kb_path}: {e}")

def run_batch():
    parser = argparse.ArgumentParser(
        description="Roda extrações em lote de forma incremental e resumível.",
        usage="python batch_extract.py [CAMINHO_JSON] [CAMINHO_PASTA_PDFS]"
    )
    parser.add_argument(
        "json_file", 
        help="Caminho para o arquivo JSON de requisições (ex: ./example_requests.json)"
    )
    parser.add_argument(
        "pdf_folder", 
        help="Caminho para a PASTA onde os PDFs/Imagens estão (ex: ./meus_pdfs/)"
    )
    # Argumento de KB opcional
    parser.add_argument(
        "-kb", "--kb_path",
        default=KB_FILENAME,
        help=f"Caminho para o arquivo de Knowledge Base (padrão: {KB_FILENAME})"
    )
    args = parser.parse_args()

    # --- 1. Inicializar Extrator ---
    log.info("Iniciando SmartExtractor...")
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY não encontrada. Defina no .env ou via 'export'.")
        return

    extractor = SmartExtractor(api_key=OPENAI_API_KEY)
    load_kb(extractor, args.kb_path)

    # --- 2. Validar Caminhos ---
    if not os.path.exists(args.json_file):
        log.error(f"Arquivo JSON de entrada não encontrado: {args.json_file}")
        return
    if not os.path.exists(args.pdf_folder) or not os.path.isdir(args.pdf_folder):
        log.error(f"Pasta de PDFs não encontrada: {args.pdf_folder}")
        return
        
    log.info(f"Arquivo JSON: {args.json_file}")
    log.info(f"Pasta de PDFs: {args.pdf_folder}")

    # --- 3. Carregar Arquivos de Entrada e Saída ---
    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            all_jobs = json.load(f)
        if not isinstance(all_jobs, list):
            raise ValueError("Arquivo de entrada deve ser uma lista JSON.")
    except Exception as e:
        log.error(f"Falha ao ler arquivo de entrada {args.json_file}: {e}")
        return

    processed_results = []
    processed_paths = set()
    output_path = OUTPUT_FILENAME # Saída sempre no diretório atual
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_results = json.load(f)
            # Usamos uma chave especial para saber qual PDF já foi processado
            processed_paths = {res.get('__pdf_path_full') for res in processed_results}
            log.info(f"Encontrados {len(processed_results)} resultados já processados em {output_path}.")
        except Exception as e:
            log.warning(f"Arquivo de saída {output_path} existe mas não pôde ser lido. Começando do zero. Erro: {e}")
            processed_results = []

    # --- 4. Criar Lista de Requisições ---
    requests_list = []
    jobs_to_run_map = {}
    
    for job in all_jobs:
        pdf_filename = job.get('pdf_path') # Ex: "oab_1.pdf"
        if not pdf_filename:
            log.warning(f"Job sem 'pdf_path' no JSON. Pulando: {job}")
            continue
            
        # --- LÓGICA PRINCIPAL MODIFICADA ---
        # Cria o caminho completo
        full_pdf_path = os.path.join(args.pdf_folder, pdf_filename)
        
        if not os.path.exists(full_pdf_path):
            log.warning(f"PDF não encontrado no caminho: {full_pdf_path}. Pulando.")
            continue
            
        # Verifica se já foi processado (usando o caminho completo)
        if full_pdf_path not in processed_paths:
            req = ExtractRequest(
                label=job.get('label'),
                extraction_schema=job.get('extraction_schema'),
                pdf_path=full_pdf_path # Usa o caminho completo
            )
            requests_list.append(req)
            jobs_to_run_map[full_pdf_path] = job # Salva o job original

    if not requests_list:
        log.info("Todos os jobs já foram processados. Encerrando.")
        return

    total_new = len(requests_list)
    total_all = len(all_jobs)
    log.info(f"Processando {total_new} novos jobs de um total de {total_all}.")

    # --- 5. EXECUTAR E SALVAR INCREMENTALMENTE ---
    current_done_count = len(processed_results) # Quantos já estavam prontos

    try:
        results_generator = extractor.extract_batch(requests_list)
        start_time = time.time()

        for i, result in enumerate(results_generator):
            progress_index = current_done_count + i + 1
            
            # Pega o caminho completo do PDF que acabou de ser processado
            current_pdf_path_full = requests_list[i].pdf_path
            # Pega o nome original do arquivo
            current_pdf_filename = os.path.basename(current_pdf_path_full)
            
            # --- FEEDBACK VISUAL ---
            print("\n" + "="*70)
            print(f"PROCESSADO: [ {progress_index} / {total_all} ] - {current_pdf_filename}")
            print("="*70)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("="*70 + "\n")
            # --- FIM DO FEEDBACK ---

            # Adiciona metadados para a lógica de resumo e referência
            result['__pdf_path_full'] = current_pdf_path_full
            result['__pdf_filename'] = current_pdf_filename
            result['__label'] = requests_list[i].label
            
            processed_results.append(result)

            # --- SALVAMENTO INCREMENTAL ---
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                log.error(f"FALHA AO SALVAR INCREMENTALMENTE em {output_path}: {e}")

        time_taken = time.time() - start_time
        log.info(f"Processamento dos {total_new} novos itens concluído.")
        log.info(f"Tempo total: {time_taken:.2f}s")
        if total_new > 0:
            log.info(f"Tempo médio: {time_taken / total_new:.2f}s por item.")

    except KeyboardInterrupt:
        log.warning("\nProcessamento interrompido pelo usuário (Ctrl+C).")
        log.warning(f"Os resultados processados até agora estão salvos em {output_path}.")
    except Exception as e:
        log.error(f"Erro fatal durante o processamento em lote: {e}", exc_info=True)
    finally:
        log.info("Salvando Knowledge Base...")
        save_kb(extractor, args.kb_path)
        log.info(f"Script finalizado. Resultados salvos em {output_path}.")

if __name__ == "__main__":
    run_batch()