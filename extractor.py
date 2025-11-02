import pdfplumber
import json
import re
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import hashlib


@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    confidence: float
    method: str  # 'heuristic', 'llm', 'hybrid'
    time_taken: float
    cost: float


class SmartExtractor:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Caches em memória
        self.schema_cache = {}  # {label: set(campos)}
        self.example_cache = {}  # {label: [exemplos com embeddings]}
        self.heuristic_patterns = {}  # {label: {campo: [patterns]}}
        self.layout_info = {}  # {label: info sobre estrutura}
        
        # Custos do GPT-4o-mini (por 1M tokens)
        self.INPUT_COST = 0.150 / 1_000_000  # $0.150 per 1M input tokens
        self.OUTPUT_COST = 0.600 / 1_000_000  # $0.600 per 1M output tokens
    
    def extract(self, label: str, extraction_schema: Dict[str, str], 
                pdf_path: str) -> Dict[str, Any]:
        """Método principal de extração"""
        start_time = time.time()
        
        # 1. Extrai texto do PDF
        text = self._extract_text_from_pdf(pdf_path)
        
        # 2. Tenta heurísticas primeiro (formulários fixos)
        if self._is_fixed_form(label, text):
            result = self._extract_with_heuristics(label, text, extraction_schema)
            if result and self._calculate_confidence(result, extraction_schema) > 0.9:
                print(f"✓ Extraído com heurísticas ({time.time() - start_time:.2f}s)")
                return result
        
        # 3. Usa LLM otimizado
        result = self._extract_with_llm(label, text, extraction_schema)
        
        # 4. Aprende para próximas vezes
        self._learn_from_extraction(label, text, extraction_schema, result)
        
        print(f"✓ Extraído com LLM ({time.time() - start_time:.2f}s)")
        return result
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai texto do PDF mantendo estrutura"""
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]  # Apenas 1 página conforme enunciado
            text = page.extract_text()
        return text
    
    def _is_fixed_form(self, label: str, text: str) -> bool:
        """Identifica se é formulário com estrutura fixa"""
        if label not in self.layout_info:
            # Primeira vez vendo este label
            # Heurística: formulários têm muitos ":" e estrutura clara
            colon_count = text.count(':')
            line_count = len(text.split('\n'))
            if line_count > 0:
                colon_ratio = colon_count / line_count
                return colon_ratio > 0.3  # Mais de 30% das linhas têm ":"
        
        return self.layout_info.get(label, {}).get('is_fixed_form', False)
    
    def _extract_with_heuristics(self, label: str, text: str, 
                                 schema: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extração rápida usando regex e patterns"""
        result = {}
        
        # Patterns comuns
        patterns = {
            'cpf': r'\d{3}\.\d{3}\.\d{3}-\d{2}',
            'cnpj': r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}',
            'telefone': r'\(\d{2}\)\s?\d{4,5}-?\d{4}',
            'cep': r'\d{5}-?\d{3}',
            'data': r'\d{2}/\d{2}/\d{4}',
            'numero': r'\d+',
        }
        
        for campo, descricao in schema.items():
            # Tenta patterns conhecidos
            if 'cpf' in campo.lower():
                match = re.search(patterns['cpf'], text)
                if match:
                    result[campo] = match.group()
                    continue
            
            if 'cnpj' in campo.lower():
                match = re.search(patterns['cnpj'], text)
                if match:
                    result[campo] = match.group()
                    continue
            
            if 'telefone' in campo.lower():
                match = re.search(patterns['telefone'], text)
                if match:
                    result[campo] = match.group()
                    continue
            
            # Busca por "campo: valor" (formulários fixos)
            campo_pattern = rf'{campo}[:\s]+([^\n]+)'
            match = re.search(campo_pattern, text, re.IGNORECASE)
            if match:
                result[campo] = match.group(1).strip()
                continue
            
            # Patterns aprendidos
            if label in self.heuristic_patterns:
                if campo in self.heuristic_patterns[label]:
                    for pattern in self.heuristic_patterns[label][campo]:
                        match = re.search(pattern, text)
                        if match:
                            result[campo] = match.group(1) if match.groups() else match.group()
                            break
        
        return result if result else None
    
    def _extract_with_llm(self, label: str, text: str, 
                         schema: Dict[str, str]) -> Dict[str, Any]:
        """Extração usando LLM com otimizações"""
        
        # 1. Chunking semântico se texto for grande
        if len(text) > 3000:
            text = self._get_relevant_chunks(text, schema)
        
        # 2. Busca exemplos similares para few-shot
        examples = self._get_similar_examples(label, text, k=2)
        
        # 3. Constrói prompt otimizado
        prompt = self._build_prompt(label, text, schema, examples)
        
        # 4. Chama LLM
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Você é um especialista em extração de dados de documentos. Retorne APENAS um JSON válido, sem markdown ou explicações."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Baixa temperatura para consistência
            response_format={"type": "json_object"}
        )
        
        # Calcula custo
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens * self.INPUT_COST) + (output_tokens * self.OUTPUT_COST)
        print(f"  💰 Custo: ${cost:.6f} ({input_tokens} in + {output_tokens} out tokens)")
        
        # Parse resposta
        result_text = response.choices[0].message.content
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback: tenta extrair JSON do texto
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {}
        
        # Garante que todos os campos pedidos estão presentes
        for campo in schema.keys():
            if campo not in result:
                result[campo] = None
        
        return result
    
    def _get_relevant_chunks(self, text: str, schema: Dict[str, str]) -> str:
        """Chunking semântico para reduzir tokens"""
        # Divide em parágrafos/seções
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(chunks) <= 3:
            return text
        
        # Embedding da query (campos que precisa extrair)
        query = " ".join([f"{k}: {v}" for k, v in schema.items()])
        query_emb = self.embedding_model.encode([query])[0]
        
        # Embeddings dos chunks
        chunk_embs = self.embedding_model.encode(chunks)
        
        # Top-5 chunks mais relevantes
        similarities = cosine_similarity([query_emb], chunk_embs)[0]
        top_indices = similarities.argsort()[-5:][::-1]
        
        relevant_chunks = [chunks[i] for i in sorted(top_indices)]
        return "\n\n".join(relevant_chunks)
    
    def _get_similar_examples(self, label: str, text: str, k: int = 2) -> List[Dict]:
        """Busca exemplos similares para few-shot learning"""
        if label not in self.example_cache or not self.example_cache[label]:
            return []
        
        # Embedding do documento atual (primeiros 500 chars)
        doc_emb = self.embedding_model.encode([text[:500]])[0]
        
        # Calcula similaridade com exemplos
        examples = self.example_cache[label]
        similarities = []
        
        for ex in examples:
            sim = cosine_similarity([doc_emb], [ex['embedding']])[0][0]
            similarities.append((ex, sim))
        
        # Retorna top-k mais similares
        top_examples = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return [ex[0]['data'] for ex in top_examples]
    
    def _build_prompt(self, label: str, text: str, schema: Dict[str, str], 
                     examples: List[Dict]) -> str:
        """Constrói prompt otimizado"""
        
        prompt_parts = []
        
        # Contexto sobre o tipo de documento
        prompt_parts.append(f"Documento do tipo: {label}")
        
        # Schema completo conhecido (se houver)
        if label in self.schema_cache and len(self.schema_cache[label]) > len(schema):
            prompt_parts.append(f"\nCampos tipicamente presentes neste tipo de documento: {', '.join(self.schema_cache[label])}")
        
        # Few-shot examples
        if examples:
            prompt_parts.append("\n## Exemplos de extrações similares:")
            for i, ex in enumerate(examples, 1):
                prompt_parts.append(f"\nExemplo {i}:")
                prompt_parts.append(json.dumps(ex, indent=2, ensure_ascii=False))
        
        # Schema a extrair
        prompt_parts.append("\n## Campos a extrair:")
        for campo, descricao in schema.items():
            prompt_parts.append(f"- {campo}: {descricao}")
        
        # Instruções
        prompt_parts.append("\n## Instruções:")
        prompt_parts.append("1. Extraia APENAS os campos solicitados")
        prompt_parts.append("2. Se um campo não existir no documento, retorne null")
        prompt_parts.append("3. Mantenha formatação original (não normalize)")
        prompt_parts.append("4. Retorne JSON válido")
        
        # Texto do documento
        prompt_parts.append("\n## Documento:")
        prompt_parts.append(text)
        
        prompt_parts.append("\n## Resposta (JSON):")
        
        return "\n".join(prompt_parts)
    
    def _calculate_confidence(self, result: Dict, schema: Dict[str, str]) -> float:
        """Calcula confiança da extração"""
        if not result:
            return 0.0
        
        # Conta quantos campos foram extraídos
        extracted = sum(1 for v in result.values() if v is not None)
        total = len(schema)
        
        return extracted / total if total > 0 else 0.0
    
    def _learn_from_extraction(self, label: str, text: str, 
                               schema: Dict[str, str], result: Dict[str, Any]):
        """Aprende com a extração para melhorar futuras"""
        
        # Atualiza schema cache
        if label not in self.schema_cache:
            self.schema_cache[label] = set()
        self.schema_cache[label].update(schema.keys())
        
        # Salva exemplo para few-shot
        if label not in self.example_cache:
            self.example_cache[label] = []
        
        # Limita a 50 exemplos por label
        if len(self.example_cache[label]) < 50:
            text_sample = text[:500]
            embedding = self.embedding_model.encode([text_sample])[0]
            
            self.example_cache[label].append({
                'data': result,
                'schema': schema,
                'embedding': embedding
            })
        
        # Aprende patterns para heurísticas
        if label not in self.heuristic_patterns:
            self.heuristic_patterns[label] = {}
        
        for campo, valor in result.items():
            if valor and isinstance(valor, str):
                # Tenta identificar pattern no texto
                escaped_valor = re.escape(valor)
                pattern = rf'([^\n]*{escaped_valor}[^\n]*)'
                match = re.search(pattern, text)
                if match:
                    if campo not in self.heuristic_patterns[label]:
                        self.heuristic_patterns[label][campo] = []
                    # Salva contexto ao redor do valor
                    self.heuristic_patterns[label][campo].append(pattern)
        
        # Identifica se é formulário fixo
        if label not in self.layout_info:
            self.layout_info[label] = {
                'is_fixed_form': self._is_fixed_form(label, text),
                'sample_count': 1
            }
        else:
            self.layout_info[label]['sample_count'] += 1