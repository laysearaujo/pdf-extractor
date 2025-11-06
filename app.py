from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import os
import tempfile
import time
import uuid
import logging
import shutil
import threading # Necessário para rodar em background
from extractor import SmartExtractor, ExtractRequest

app = Flask(__name__)
CORS(app)

# --- CONFIGURAÇÃO ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) 
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- INICIALIZAÇÃO DO EXTRATOR ---
print("Iniciando SmartExtractor...")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if not OPENAI_API_KEY:
             raise ValueError("OPENAI_API_KEY não encontrada. Defina no .env ou via 'export'.")
        log.info("Chave OPENAI_API_KEY carregada do .env")
    except ImportError:
        raise ValueError("OPENAI_API_KEY não encontrada e python-dotenv não instalado.")

extractor = SmartExtractor(api_key=OPENAI_API_KEY)
# Tenta carregar um KB existente
kb_path = "knowledge_base.json"
if os.path.exists(kb_path):
    extractor.import_kb(kb_path)

print("SmartExtractor pronto.")

# --- ARMAZENAMENTO DE JOBS EM MEMÓRIA ---
# Este dicionário global guardará o progresso dos jobs
BATCH_JOBS = {}
# Um "Lock" é essencial para evitar que duas requisições
# mexam no dicionário ao mesmo tempo (thread-safety)
job_lock = threading.Lock()
# -----------------------------------

def process_batch_in_background(job_id: str, requests_list: list, kb_path: str, temp_dir: str):
    """
    Esta é a função que roda na thread em segundo plano.
    Ela usa o generator 'extract_batch' e atualiza o 'placar' (BATCH_JOBS).
    """
    total_files = len(requests_list)
    try:
        log.info(f"[Job {job_id}] Processamento em background iniciado para {total_files} arquivos.")
        
        # Pega o generator (ele ainda não executou)
        results_generator = extractor.extract_batch(requests_list)
        
        processed_count = 0
        
        # Itera sobre o generator (AQUI a extração acontece, uma por uma)
        for result in results_generator:
            processed_count += 1
            log.info(f"[Job {job_id}] Processado {processed_count}/{total_files}...")
            
            # Atualiza o "placar" global de forma segura
            with job_lock:
                BATCH_JOBS[job_id]["processed_count"] = processed_count
                # Salva o último resultado para o front-end ver
                BATCH_JOBS[job_id]["last_result"] = result 
                BATCH_JOBS[job_id]["results"].append(result)
                BATCH_JOBS[job_id]["status"] = "running"
        
        # Se o loop terminar, o job está completo
        log.info(f"[Job {job_id}] Processamento concluído.")
        with job_lock:
            BATCH_JOBS[job_id]["status"] = "complete"
            
        # Salva o KB após o lote
        extractor.export_kb(kb_path)

    except Exception as e:
        log.error(f"[Job {job_id}] Falha na thread do job: {e}", exc_info=True)
        with job_lock:
            BATCH_JOBS[job_id]["status"] = "failed"
            BATCH_JOBS[job_id]["error"] = str(e)
    finally:
        # Limpa os arquivos temporários DEPOIS que a thread terminar
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                log.info(f"[Job {job_id}] Diretoria temporária {temp_dir} limpa.")
            except Exception as e:
                log.error(f"[Job {job_id}] Falha ao limpar diretoria temporária {temp_dir}: {e}")

# -----------------------------------

@app.route('/')
def index():
    """Interface web (lendo de templates/index.html)"""
    return render_template('index.html')

@app.route('/api/extract', methods=['POST'])
def extract_endpoint():
    """Endpoint para extração única (usado pela UI) - Sem mudanças"""
    start_time = time.time()
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'PDF não fornecido'}), 400
        
        label = request.form.get('label')
        schema_str = request.form.get('extraction_schema')
        pdf_file = request.files['pdf']
        
        if not label or not schema_str:
            return jsonify({'error': 'label e extraction_schema são obrigatórios'}), 400
        
        try:
            extraction_schema = json.loads(schema_str)
        except json.JSONDecodeError:
            return jsonify({'error': 'extraction_schema deve ser um JSON válido'}), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            pdf_file.save(temp.name)
            temp_path = temp.name

        try:
            result = extractor.extract(label, extraction_schema, temp_path)
            extractor.export_kb(kb_path)
        finally:
            os.remove(temp_path)
        
        time_taken = time.time() - start_time
        
        return jsonify({
            'data': result,
            'time_taken': round(time_taken, 3),
            'method': 'smart_extraction'
        })
        
    except Exception as e:
        log.error(f"Erro no /api/extract: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# --- MODIFICADO: Endpoint de Lote agora inicia um Job ---
@app.route('/api/batch_upload', methods=['POST'])
def batch_upload_endpoint():
    """
    Inicia um job de processamento em lote.
    Não processa; apenas salva os arquivos e inicia uma thread.
    Retorna imediatamente um job_id.
    """
    if 'request_json' not in request.files:
        return jsonify({'error': 'Ficheiro request_json não encontrado'}), 400
    if 'files' not in request.files:
         return jsonify({'error': 'Nenhum ficheiro (files) enviado'}), 400

    json_file = request.files['request_json']
    pdf_files = request.files.getlist('files')

    # Cria uma diretoria temporária única
    request_id = str(uuid.uuid4()) # Este será o nosso Job ID
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], request_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path_map = {}
    
    try:
        # 1. Salva os arquivos
        for f in pdf_files:
            filename = f.filename
            safe_path = os.path.join(temp_dir, filename)
            f.save(safe_path)
            file_path_map[filename] = safe_path

        # 2. Lê o JSON de pedidos
        try:
            request_jobs = json.load(json_file)
        except Exception as e:
            return jsonify({'error': f'JSON de pedidos inválido: {e}'}), 400

        # 3. Cria a lista de ExtractRequest
        requests_list = []
        for job in request_jobs:
            pdf_path_from_json = job.get('pdf_path')
            real_pdf_path = file_path_map.get(pdf_path_from_json)
            if not real_pdf_path:
                log.warning(f"[Job {request_id}] Ficheiro '{pdf_path_from_json}' não foi enviado.")
                continue
            requests_list.append(
                ExtractRequest(
                    label=job.get('label'),
                    extraction_schema=job.get('extraction_schema'),
                    pdf_path=real_pdf_path
                )
            )

        if not requests_list:
            return jsonify({'error': 'Nenhum ficheiro válido para processar.'}), 400
            
        # 4. Registra o Job no "placar" global
        total_files = len(requests_list)
        with job_lock:
            BATCH_JOBS[request_id] = {
                "status": "starting",
                "processed_count": 0,
                "total_files": total_files,
                "results": [],
                "last_result": None,
                "error": None
            }
            
        # 5. Inicia a Thread em Background
        thread = threading.Thread(
            target=process_batch_in_background,
            args=(request_id, requests_list, kb_path, temp_dir)
        )
        thread.start() # Inicia a thread e libera esta requisição

        log.info(f"[Job {request_id}] Job iniciado. Retornando job_id para o cliente.")
        
        # 6. Retorna IMEDIATAMENTE para o front-end
        return jsonify({
            'job_id': request_id,
            'total_files': total_files,
            'message': 'Job started successfully.'
        })

    except Exception as e:
        log.exception(f"Erro ao iniciar job /api/batch_upload: {e}")
        # Limpa os arquivos se a *criação* do job falhar
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({'error': str(e)}), 500
    # Note: O 'finally' que limpava o temp_dir foi REMOVIDO daqui
    # e movido para dentro da thread 'process_batch_in_background'

# --- NOVO: Endpoint para o Front-end "perguntar" o progresso ---
@app.route('/api/batch_status/<job_id>', methods=['GET'])
def batch_status_endpoint(job_id):
    """
    Retorna o status atual de um job em processamento.
    """
    with job_lock:
        job_info = BATCH_JOBS.get(job_id)

    if not job_info:
        return jsonify({'error': 'Job not found'}), 404
    
    # Retorna a informação de status atual
    return jsonify(job_info)

# --- NOVO: Endpoint para limpar um job da memória ---
@app.route('/api/batch_cleanup/<job_id>', methods=['POST'])
def batch_cleanup_endpoint(job_id):
    """
    Remove um job finalizado da memória para economizar espaço.
    """
    with job_lock:
        if job_id in BATCH_JOBS:
            if BATCH_JOBS[job_id]["status"] in ("complete", "failed"):
                del BATCH_JOBS[job_id]
                log.info(f"[Job {job_id}] Job limpo da memória.")
                return jsonify({"status": "cleaned"}), 200
            else:
                return jsonify({"error": "Job still running"}), 400
        else:
            return jsonify({"error": "Job not found"}), 404
            
# --- Outros Endpoints (sem mudanças) ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'extractor_ready': 'true'})

@app.route('/api/get_kb', methods=['GET'])
def get_kb():
    return jsonify({
        'metadata': extractor.label_metadata,
        'kb': extractor.KB
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)