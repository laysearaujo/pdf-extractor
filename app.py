from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from pathlib import Path
from extractor import SmartExtractor
import time

app = Flask(__name__)
CORS(app)

# Inicializa extrator
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")

extractor = SmartExtractor(OPENAI_API_KEY)

# Template HTML da interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart PDF Extractor - ENTER AI Fellowship</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 40px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input[type="text"], textarea, input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border 0.3s;
        }
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            min-height: 150px;
            font-family: 'Courier New', monospace;
            resize: vertical;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result-box {
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        .result-box.show {
            display: block;
        }
        .result-box h3 {
            color: #28a745;
            margin-bottom: 15px;
        }
        .result-box pre {
            background: white;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-card .label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .batch-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .example-json {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Smart PDF Extractor</h1>
            <p>Solução otimizada para extração de dados estruturados de PDFs</p>
        </div>
        
        <div class="content">
            <!-- Extração Simples -->
            <div class="section">
                <h2>📄 Extração Única</h2>
                
                <div class="input-group">
                    <label for="label">Label do Documento:</label>
                    <input type="text" id="label" placeholder="Ex: carteira_oab, contrato, fatura">
                </div>
                
                <div class="input-group">
                    <label for="schema">Schema de Extração (JSON):</label>
                    <textarea id="schema" placeholder='{"campo": "descrição do campo"}'></textarea>
                </div>
                
                <div class="input-group">
                    <label for="pdf">Arquivo PDF:</label>
                    <input type="file" id="pdf" accept=".pdf">
                </div>
                
                <button class="btn" onclick="extractSingle()">Extrair Dados</button>
                
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <p>Processando documento...</p>
                </div>
                
                <div id="result" class="result-box">
                    <h3>✅ Resultado da Extração</h3>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="label">Tempo</div>
                            <div class="value" id="time">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Método</div>
                            <div class="value" id="method">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Campos</div>
                            <div class="value" id="fields">-</div>
                        </div>
                    </div>
                    <pre id="resultJson"></pre>
                </div>
            </div>
            
            <!-- Processamento em Lote -->
            <div class="section batch-section">
                <h2>📦 Processamento em Lote</h2>
                <p style="margin-bottom: 15px; color: #666;">
                    Para processar múltiplos documentos, use o script CLI ou faça requisições à API.
                </p>
                
                <h4>Exemplo de JSON para lote:</h4>
                <div class="example-json">
[
  {
    "label": "carteira_oab",
    "extraction_schema": {
      "nome": "Nome do profissional",
      "inscricao": "Número de inscrição"
    },
    "pdf_path": "oab_1.pdf"
  },
  {
    "label": "contrato",
    "extraction_schema": {
      "partes": "Partes do contrato",
      "valor": "Valor do contrato"
    },
    "pdf_path": "contrato_1.pdf"
  }
]
                </div>
                
                <h4 style="margin-top: 20px;">Endpoint da API:</h4>
                <code style="background: #f0f0f0; padding: 10px; display: block; border-radius: 5px;">
                    POST /api/extract
                </code>
            </div>
        </div>
    </div>
    
    <script>
        async function extractSingle() {
            const label = document.getElementById('label').value;
            const schemaText = document.getElementById('schema').value;
            const pdfFile = document.getElementById('pdf').files[0];
            
            if (!label || !schemaText || !pdfFile) {
                alert('Por favor, preencha todos os campos!');
                return;
            }
            
            let schema;
            try {
                schema = JSON.parse(schemaText);
            } catch (e) {
                alert('Schema JSON inválido!');
                return;
            }
            
            const formData = new FormData();
            formData.append('label', label);
            formData.append('extraction_schema', JSON.stringify(schema));
            formData.append('pdf', pdfFile);
            
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            
            const startTime = Date.now();
            
            try {
                const response = await fetch('/api/extract', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                const timeTaken = ((Date.now() - startTime) / 1000).toFixed(2);
                
                document.getElementById('loading').classList.remove('show');
                document.getElementById('result').classList.add('show');
                
                document.getElementById('time').textContent = timeTaken + 's';
                document.getElementById('method').textContent = data.method || 'LLM';
                document.getElementById('fields').textContent = Object.keys(data.data || {}).length;
                document.getElementById('resultJson').textContent = JSON.stringify(data.data, null, 2);
                
            } catch (error) {
                document.getElementById('loading').classList.remove('show');
                alert('Erro ao processar: ' + error.message);
            }
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Interface web"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/extract', methods=['POST'])
def extract_endpoint():
    """
    Endpoint para extração única
    Aceita form-data com: label, extraction_schema (JSON string), pdf (file)
    """
    try:
        start_time = time.time()
        
        # Valida requisição
        if 'pdf' not in request.files:
            return jsonify({'error': 'PDF não fornecido'}), 400
        
        label = request.form.get('label')
        schema_str = request.form.get('extraction_schema')
        pdf_file = request.files['pdf']
        
        if not label or not schema_str:
            return jsonify({'error': 'label e extraction_schema são obrigatórios'}), 400
        
        # Parse schema
        try:
            extraction_schema = json.loads(schema_str)
        except json.JSONDecodeError:
            return jsonify({'error': 'extraction_schema deve ser um JSON válido'}), 400
        
        # Salva PDF temporariamente
        temp_path = f'/tmp/{pdf_file.filename}'
        pdf_file.save(temp_path)
        
        # Extrai dados
        result = extractor.extract(label, extraction_schema, temp_path)
        
        # Remove arquivo temporário
        os.remove(temp_path)
        
        time_taken = time.time() - start_time
        
        return jsonify({
            'data': result,
            'time_taken': round(time_taken, 3),
            'method': 'smart_extraction'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
def batch_extract_endpoint():
    """
    Endpoint para processamento em lote
    Aceita JSON array com: [{label, extraction_schema, pdf_path}]
    """
    try:
        data = request.json
        
        if not isinstance(data, list):
            return jsonify({'error': 'Esperado um array de requisições'}), 400
        
        results = []
        total_time = 0
        
        for i, item in enumerate(data):
            start_time = time.time()
            
            label = item.get('label')
            extraction_schema = item.get('extraction_schema')
            pdf_path = item.get('pdf_path')
            
            if not all([label, extraction_schema, pdf_path]):
                results.append({
                    'index': i,
                    'error': 'label, extraction_schema e pdf_path são obrigatórios'
                })
                continue
            
            # Verifica se arquivo existe
            if not os.path.exists(pdf_path):
                results.append({
                    'index': i,
                    'error': f'Arquivo não encontrado: {pdf_path}'
                })
                continue
            
            # Extrai
            try:
                result = extractor.extract(label, extraction_schema, pdf_path)
                time_taken = time.time() - start_time
                total_time += time_taken
                
                results.append({
                    'index': i,
                    'data': result,
                    'time_taken': round(time_taken, 3)
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_time': round(total_time, 3),
            'avg_time': round(total_time / len(data), 3) if data else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'extractor': 'ready'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)