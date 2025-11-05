from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import tempfile
from extractor import SmartExtractor
import time

app = Flask(__name__)
CORS(app)

# --- INICIALIZAÇÃO DO EXTRATOR ---
print("Iniciando SmartExtractor...")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não encontrada. Defina no .env ou via 'export'.")

extractor = SmartExtractor(OPENAI_API_KEY)
print("✓ SmartExtractor pronto.")
# -----------------------------------

@app.route('/')
def index():
    """Interface web (lendo de templates/index.html)"""
    return render_template('index.html')

@app.route('/api/extract', methods=['POST'])
def extract_endpoint():
    """Endpoint para extração única (usado pela UI)"""
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
        
        # Salva PDF de forma segura e temporária
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            pdf_file.save(temp.name)
            temp_path = temp.name

        try:
            # Chama o "Cérebro"
            result = extractor.extract(label, extraction_schema, temp_path)
        finally:
            os.remove(temp_path)
        
        print(result)
        time_taken = time.time() - start_time
        
        return jsonify({
            'data': result,
            'time_taken': round(time_taken, 3),
            'method': 'smart_extraction'
        })
        
    except Exception as e:
        log.error(f"Erro no /api/extract: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_extract_endpoint():
    """Endpoint para processamento em lote (usado pelo script CLI)"""
    try:
        data = request.json
        if not isinstance(data, list):
            return jsonify({'error': 'Esperado um array de requisições'}), 400
        
        results = []
        total_time = 0
        
        for i, item in enumerate(data):
            start_item = time.time()
            label = item.get('label')
            extraction_schema = item.get('extraction_schema')
            pdf_path = item.get('pdf_path')
            
            if not all([label, extraction_schema, pdf_path]):
                results.append({'index': i, 'error': 'Campos obrigatórios faltando'})
                continue
            
            if not os.path.exists(pdf_path):
                results.append({'index': i, 'error': f'Arquivo não encontrado: {pdf_path}'})
                continue
            
            try:
                result = extractor.extract(label, extraction_schema, pdf_path)
                time_taken = time.time() - start_item
                total_time += time_taken
                results.append({
                    'index': i,
                    'data': result,
                    'time_taken': round(time_taken, 3)
                })
            except Exception as e:
                results.append({'index': i, 'error': str(e)})
        
        return jsonify({
            'results': results,
            'total_time': round(total_time, 3),
            'avg_time': round(total_time / len(data), 3) if data else 0
        })
        
    except Exception as e:
        log.error(f"Erro no /api/batch: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'extractor_ready': 'true'})

# NOVO ENDPOINT: Para inspecionar o que foi aprendido
@app.route('/api/get_kb', methods=['GET'])
def get_kb():
    """Retorna o Knowledge Base (heurísticas e metadados)"""
    return jsonify({
        'metadata': extractor.label_metadata,
        'kb': extractor.KB
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)