import fitz  # PyMuPDF
import re
import json
import logging
from openai import OpenAI
# ATUALIZADO: Importar Generator
from typing import Optional, Any, List, Dict, Tuple, Generator 
from dataclasses import dataclass, field as dataclass_field
import hashlib

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Mantido conforme sua solicitação
LLM_MODEL_NAME = "gpt-5-mini"


@dataclass
class Heuristic:
    """Representa uma heurística aprendida"""
    type: str  # ANCHOR, ZONE, ANCHOR_EMPTY
    value: Any
    confidence: float = 1.0
    metadata: Dict = dataclass_field(default_factory=dict)


# Dataclass para gerir pedidos em lote
@dataclass
class ExtractRequest:
    """Representa um único pedido de extração num lote."""
    label: str
    extraction_schema: Dict[str, str]
    pdf_path: str

    _parsed_pdf: Optional[Dict] = None
    _pdf_hash: Optional[str] = None
    _result: Dict[str, Any] = dataclass_field(default_factory=dict)
    _failed_fields: List[str] = dataclass_field(default_factory=list)


class SmartExtractor:
    """
    Extrator inteligente com estratégia de Template Fixo/Variável.
    """
    
    def __init__(self, api_key: str):
        self.KB: Dict[str, Dict[str, Heuristic]] = {}
        self.label_metadata: Dict[str, Dict] = {} 
        
        self.pdf_cache: Dict[str, Dict] = {} # Cache de RESULTADO
        self.parsed_pdf_cache: Dict[str, Dict] = {} # Cache de PARSE (para velocidade)
        self.llm_client = OpenAI(api_key=api_key)
        
        self.INPUT_COST = 0.150 / 1_000_000
        self.OUTPUT_COST = 0.600 / 1_000_000
        
        self.stats = {
            'cache_hits': 0,
            'anchor_extractions': 0,
            'zone_extractions': 0,
            'llm_bootstraps': 0,
            'llm_fallbacks': 0, # Agora conta fallbacks individuais
            'total_cost': 0.0
        }
        
        log.info(f"SmartExtractor (Bootstrap LLM) inicializado com {LLM_MODEL_NAME}.")

    # ==================== HELPERS ====================
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as f:
                # Lê em blocos para não sobrecarregar a memória
                sha256_hash = hashlib.sha256()
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
                return sha256_hash.hexdigest()
        except Exception as e:
            log.warning(f"Não foi possível gerar hash para {pdf_path}: {e}")
            return ""

    def _parse_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        doc = None
        try:
            doc = fitz.open(pdf_path)
            if not doc or len(doc) == 0:
                log.warning(f"Documento PDF vazio ou inválido: {pdf_path}")
                return None
            
            page = doc[0]

            # --- DADOS PARA O LLM (com fallback) ---
            try:
                # 1. Tenta o modo "sort=True" (ideal para o LLM ver o layout)
                layout_text = page.get_text("text", sort=True)
                log.info("Extração de texto (modo 'sort=True') bem-sucedida.")
            except Exception as sort_error:
                # 2. Se a ordenação falhar, volta para o modo padrão (texto empilhado)
                log.warning(f"Extração (modo 'sort=True') falhou: {sort_error}. Voltando para o modo padrão (lógico).")
                layout_text = page.get_text("text", sort=False) 

            # print("FULL TEXT (final usado para LLM):", layout_text) # Removido para diminuir o log
            
            # --- DADOS PARA AS HEURÍSTICAS ---
            words_data = page.get_text("words")
            if not words_data:
                log.error(f"PDF não contém texto legível (sem 'words'): {pdf_path}")
                return None

            # --- OTIMIZAÇÃO: Pré-calcular palavras normalizadas UMA VEZ ---
            norm_words = []
            for w in words_data:
                norm_text = self._normalize_text(w[4])
                if norm_text:
                    norm_words.append((norm_text, fitz.Rect(w[0:4])))
            
            return {
                "doc": doc,
                "page": page,
                "words": words_data,
                "norm_words": norm_words,
                "full_text": layout_text,
                "clean_text": layout_text
            }
            
        except Exception as e:
            log.exception(f"Erro detalhado ao parsear PDF: {pdf_path}, ERRO: {e}")
            if doc:
                doc.close()
            return None
    
    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[áàâã]', 'a', text, flags=re.IGNORECASE)
        text = re.sub(r'[éê]', 'e', text, flags=re.IGNORECASE)
        text = re.sub(r'[í]', 'i', text, flags=re.IGNORECASE)
        text = re.sub(r'[óôõ]', 'o', text, flags=re.IGNORECASE)
        text = re.sub(r'[ú]', 'u', text, flags=re.IGNORECASE)
        text = re.sub(r'[ç]', 'c', text, flags=re.IGNORECASE)
        return re.sub(r'[\s_:]+', '', text.lower())

    def _search_for_normalized(self, page: fitz.Page, needle: str, norm_words: List[Tuple[str, fitz.Rect]]) -> List[fitz.Rect]:
        """
        Usa a lista pré-normalizada de palavras (norm_words) para encontrar a 'needle'.
        """
        if not needle:
            return []
        
        needle_norm = self._normalize_text(needle)
        if not needle_norm:
            return []
        
        for i in range(len(norm_words)):
            
            if norm_words[i][0] == needle_norm:
                return [norm_words[i][1]] # Encontrado (palavra única)

            if needle_norm.startswith(norm_words[i][0]):
                current_text = norm_words[i][0]
                current_rect = fitz.Rect(norm_words[i][1])
                
                for j in range(i + 1, min(i + 5, len(norm_words))):
                    if norm_words[j][1].y0 > current_rect.y1 + 5:
                        break 
                    current_text += norm_words[j][0]
                    current_rect.include_rect(norm_words[j][1])
                    if current_text == needle_norm:
                        return [current_rect] # Encontrado (múltiplas palavras)
                    if not needle_norm.startswith(current_text):
                        break 
        
        return [] # Não encontrado

    # ==================== NÍVEL 1: APLICADORES ====================
    
    def _apply_anchor_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Optional[str]:
        try:
            page = parsed_pdf["page"]
            anchor_text = heuristic.metadata.get("anchor_text")
            direction = heuristic.metadata.get("direction", "right")
            layout = heuristic.metadata.get("layout", "column") 
            multi_line = heuristic.metadata.get("multi_line", False)
            regex = heuristic.metadata.get("regex")
            
            if not anchor_text: return None

            # print('PAGINA:', page) # Removido para diminuir o log
            
            norm_words_data = parsed_pdf.get("norm_words")
            anchor_rects = self._search_for_normalized(page, anchor_text, norm_words_data)
            if not anchor_rects:
                log.warning(f"Não foi possível encontrar âncora (normalizada): '{anchor_text}'")
                return None
            
            anchor_rect = anchor_rects[0]
            page_rect = page.rect
            
            if direction == "right":
                search_rect = fitz.Rect(
                    anchor_rect.x1 + 2, 
                    anchor_rect.y0 - 2,
                    page_rect.width - 10, 
                    anchor_rect.y1 + 2
                )
            elif direction == "below":
                if multi_line:
                    height = anchor_rect.y1 + 70 
                else:
                    height = anchor_rect.y1 + 20 
                
                if layout == "column":
                    search_rect = fitz.Rect(
                        anchor_rect.x0 - 10, 
                        anchor_rect.y1 + 2,
                        anchor_rect.x1 + 300, 
                        height
                    )
                else: # layout == "full_width"
                    search_rect = fitz.Rect(
                        5, 
                        anchor_rect.y1 + 2,
                        page_rect.width - 10, 
                        height
                    )
            else:
                return None
            
            value_text = page.get_text("text", clip=search_rect)
            if not value_text: return None
            
            value_text = value_text.strip().replace(anchor_text, "").strip()
            
            if regex:
                match = re.search(regex, value_text)
                value_text = match.group(0) if match else None
            else:
                lines = [l.strip() for l in value_text.split('\n') if l.strip()]
                if not lines:
                    value_text = None
                elif multi_line:
                    value_text = "\n".join(lines)
                else:
                    value_text = lines[0]
            
            if value_text:
                preview = value_text.replace('\n', ' ')
                preview = preview[:50] + "..." if len(preview) > 50 else preview
                log.info(f"Âncora: '{anchor_text}' ({direction}/{layout}) → '{preview}'")
                return value_text
            
            return None
        except Exception as e:
            log.warning(f"Erro ao aplicar heurística de âncora: {e}")
            return None
    
    def _apply_zone_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Optional[str]:
        try:
            page = parsed_pdf["page"]
            zone_coords = heuristic.value
            if not zone_coords or len(zone_coords) != 4: return None
            
            zone_rect = fitz.Rect(zone_coords)
            value_text = page.get_text("text", clip=zone_rect)
            
            if value_text and value_text.strip():
                value_text = value_text.strip()
                log.info(f"Zona: {zone_coords} → '{value_text}'")
                return value_text
            
            return None
        except Exception as e:
            log.warning(f"Erro ao aplicar heurística de zona: {e}")
            return None

    def _apply_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Tuple[Optional[str], bool]:
        value = None
        
        if heuristic.type == "ANCHOR":
            self.stats['anchor_extractions'] += 1
            value = self._apply_anchor_heuristic(heuristic, parsed_pdf)
        elif heuristic.type == "ZONE":
            self.stats['zone_extractions'] += 1
            value = self._apply_zone_heuristic(heuristic, parsed_pdf)

        elif heuristic.type == "ANCHOR_EMPTY":
            log.info(f"Aplicando heurística ANCHOR_EMPTY (esperado Nulo)")
            anchor_text = heuristic.metadata.get("anchor_text")
            if not anchor_text: 
                return None, False 
            
            # Tenta aplicar a âncora normalmente
            value = self._apply_anchor_heuristic(heuristic, parsed_pdf) 
            
            if value:
                # SUCESSO: Auto-correção. O campo não é mais nulo.
                log.info(f"AUTOCORREÇÃO: ANCHOR_EMPTY encontrou um valor! '{value}'")
                return value, True 
            
            # FALHA: A heurística esperava Nulo, mas não pôde confirmar.
            # Isso pode significar que a âncora mudou OU que o valor agora existe
            # e a âncora antiga não o encontrou.
            # Em ambos os casos, deve falhar e ir para o LLM.
            log.info(f"ANCHOR_EMPTY falhou em confirmar Nulo. Tratando como falha para acionar o LLM.")
            return None, False

        else:
            return None, False

        if value is not None:
            return value, True
        else:
            return None, False

    # ==================== NÍVEL 2: APRENDIZADO NLP (FALLBACK) ====================
    
    def _learn_from_anchor(self, parsed_pdf: Dict, field: str, description: str) -> Tuple[Optional[str], Optional[Heuristic], bool]:
        page = parsed_pdf["page"]
        page_rect = page.rect
        norm_words_data = parsed_pdf.get("norm_words")
        
        anchor_candidates = list(set([
            field, field.replace("_", " "), field.replace("_", " ").title(), 
            field.upper(), description, description.upper()
        ]))

        log.info(f"  🔍(NLP) Procurando âncoras: {anchor_candidates[:3]}...")
        
        for anchor in anchor_candidates:
            if not anchor: continue
            
            rects = self._search_for_normalized(page, anchor, norm_words_data)
            if not rects: continue
            
            anchor_rect = rects[0]
            
            # Tenta à DIREITA
            search_right = fitz.Rect(anchor_rect.x1 + 2, anchor_rect.y0 - 2, page_rect.width - 10, anchor_rect.y1 + 2)
            value_right = page.get_text("text", clip=search_right)
            
            if value_right:
                value_right = value_right.strip().replace(anchor, "").strip()
                lines = [l.strip() for l in value_right.split('\n') if l.strip()]
                if lines and len(lines[0]) > 0:
                    value = "\n".join(lines)
                    heuristic = Heuristic(type="ANCHOR", value=None, confidence=0.9, metadata={"anchor_text": anchor, "direction": "right", "regex": None})
                    log.info(f"(NLP) ÂNCORA (direita): '{anchor}' → '{value[:50]}...'")
                    return value, heuristic, True
            
            # Tenta ABAIXO
            search_below = fitz.Rect(
                anchor_rect.x0 - 10, anchor_rect.y1 + 2, 
                anchor_rect.x1 + 300, anchor_rect.y1 + 20
            )
            value_below = page.get_text("text", clip=search_below)
            if value_below:
                value_below = value_below.strip()
                lines = [l.strip() for l in value_below.split('\n') if l.strip()]
                if lines:
                    value = "\n".join(lines)
                    heuristic = Heuristic(type="ANCHOR", value=None, confidence=0.85, metadata={"anchor_text": anchor, "direction": "below", "regex": None})
                    log.info(f"(NLP) ÂNCORA (abaixo): '{anchor}' → '{value[:50].replace(chr(10), ' ')}...'")
                    return value, heuristic, True
            
            log.info(f"(NLP) ÂNCORA encontrada ('{anchor}') mas valor está VAZIO.")
            heuristic = Heuristic(
                type="ANCHOR_EMPTY", value=None, confidence=0.8,
                metadata={"anchor_text": anchor, "direction": "right"} 
            )
            return None, heuristic, True

        return None, None, False

    # ==================== NÍVEL 3: APRENDIZADO LLM ====================
    
    def _call_llm(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        try:
            log.info(f"Chamando LLM... (JSON Mode: {json_mode})")
            
            model_params = {
                "model": LLM_MODEL_NAME, "messages": [
                    {"role": "system", "content": "Você é um assistente especialista em extração de dados de documentos. Siga as instruções de formato da resposta com precisão."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1.0 # Mantido em 1.0
            }
            if json_mode:
                model_params["response_format"] = {"type": "json_object"}

            response = self.llm_client.chat.completions.create(**model_params)
            log.info(f"Terminado LLM... (JSON Mode: {json_mode})")
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * self.INPUT_COST) + (output_tokens * self.OUTPUT_COST)
            self.stats['total_cost'] += cost
            
            log.info(f"Custo: ${cost:.6f} ({input_tokens} in + {output_tokens} out)")
            
            content = response.choices[0].message.content
            return content.strip() if content else None
            
        except Exception as e:
            log.error(f"Erro ao chamar LLM: {e}")
            return None
    
    def _derive_heuristic_for_value(self, parsed_pdf: Dict, field: str, value: str) -> Optional[Heuristic]:
        page = parsed_pdf["page"]
        norm_words_data = parsed_pdf.get("norm_words")
        if not value: return None
        
        # --- 1. Encontra a Localização do Valor ---
        clean_value = value.strip().replace(',', ' ').replace('\n', ' ')
        value_parts = clean_value.split()
        if len(value_parts) == 0:
            return None
        
        value_for_search = " ".join(value_parts[0:3]) # Tenta 3 palavras
        value_rects = self._search_for_normalized(page, value_for_search, norm_words_data)
        
        if not value_rects:
            value_for_search = " ".join(value_parts[0:1]) # Fallback: 1 palavra
            value_rects = self._search_for_normalized(page, value_for_search, norm_words_data)
            
            if not value_rects:
                log.warning(f"Não foi possível encontrar o texto (normalizado) '{value_for_search}' no PDF para derivar heurística.")
                return None
        
        rect = value_rects[0]
        
        # --- 2. TENTA DERIVAR UMA ÂNCORA ---
        anchor_rect_above = fitz.Rect(rect.x0 - 50, max(0, rect.y0 - 50), rect.x1 + 50, rect.y0 - 2)
        anchor_text_above = page.get_text("text", clip=anchor_rect_above).strip()
        if anchor_text_above:
            anchor = anchor_text_above.split('\n')[-1].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"Heurística ÂNCORA (acima) derivada: '{anchor}' → '{value[:20]}...'")
                return Heuristic(
                    type="ANCHOR", value=None, confidence=0.9,
                    metadata={"anchor_text": anchor, "direction": "below"}
                )

        anchor_rect_left = fitz.Rect(max(0, rect.x0 - 300), rect.y0 - 5, rect.x0 - 2, rect.y1 + 5)
        anchor_text_left = page.get_text("text", clip=anchor_rect_left).strip()
        if anchor_text_left:
            anchor = anchor_text_left.split('\n')[-1].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"Heurística ÂNCORA (esquerda) derivada: '{anchor}' → '{value[:20]}...'")
                return Heuristic(
                    type="ANCHOR", value=None, confidence=0.9,
                    metadata={"anchor_text": anchor, "direction": "right"}
                )
        
        # --- 3. FALLBACK: CRIAR UMA ZONA "HORIZONTAL SLICE" ---
        page_rect = page.rect
        y0 = max(0, rect.y0 - 5)
        y1 = min(page_rect.height - 2, rect.y1 + 5) 
        x0 = 5.0 # Margem esquerda
        x1 = page_rect.width - 5.0 # Margem direita

        if '\n' in value or len(clean_value) > 80:
            y1 = min(page_rect.height - 2, rect.y1 + 70) 
            log.info("Heurística ZONA (multi-linha) detectada.")
        
        zone_coords = [x0, y0, x1, y1]
        log.info(f"FALLBACK: Nenhuma âncora encontrada. Heurística ZONA 'Horizontal Slice' salva: {zone_coords}")
        
        return Heuristic(type="ZONE", value=zone_coords, confidence=0.7, metadata={})

    def _bootstrap_new_label_with_llm(self, label: str, extraction_schema: Dict[str, str], parsed_pdf: Dict) -> Dict[str, Any]:
        log.info(f"(LLM) Bootstrap: Label novo '{label}'. Chamando LLM para dados e classificação.")
        self.stats['llm_bootstraps'] += 1
        
        fields_json_str = json.dumps(extraction_schema, indent=2, ensure_ascii=False)
        text = parsed_pdf["full_text"] 
        
        prompt = f"""Você é um especialista em extração de dados. Analise o documento e retorne um único objeto JSON.
        O JSON deve ter DUAS chaves principais:
        1. "template_fixo": true se for um template fixo (layout previsível como RG, OAB), false se for variável (nota fiscal, tela de app).
        2. "fields": Um objeto JSON simples contendo os dados extraídos (chave: valor).
        3. Validação: Caso a chave não faça sentido com o label retorne null.

        Campos para extrair:
        {fields_json_str}

        Texto do Documento (com Layout):
        ---
        {text}
        ---
        Retorne APENAS o objeto JSON completo.
        """
        
        # print(text) # Removido para diminuir o log
        response_str = self._call_llm(prompt, json_mode=True)
        # print(f"response LLM: {response_str}") # Removido para diminuir o log
        
        if not response_str:
            log.error("(LLM) Bootstrap falhou. Nenhuma resposta do LLM.")
            return {field: None for field in extraction_schema.keys()}
        
        try:
            response_json = json.loads(response_str)
            
            is_fixed = response_json.get('template_fixo', True)
            self.label_metadata[label] = {'template_fixo': is_fixed}
            log.info(f"Template classificado como: {'FIXO' if is_fixed else 'VARIÁVEL'}")

            final_data_clean = {} 
            self.KB[label] = {}
            
            fields_data = response_json.get("fields", {})
            for field in extraction_schema.keys():
                
                value_str_raw = str(fields_data.get(field)).strip() if fields_data.get(field) else None
                
                if not value_str_raw or value_str_raw.lower() == 'null':
                    log.info(f"(LLM) Bootstrap: '{field}' → NÃO ENCONTRADO")
                    final_data_clean[field] = None
                    _, heuristic, found = self._learn_from_anchor(parsed_pdf, field, extraction_schema[field])
                    if found and heuristic and is_fixed:
                        self.KB[label][field] = heuristic
                        log.info(f"Heurística NLP ('{heuristic.type}') salva para campo Nulo!")
                    continue

                log.info(f"(LLM) Bootstrap: '{field}' → '{value_str_raw[:50].replace(chr(10), ' ')}...'")
                
                heuristic = self._derive_heuristic_for_value(parsed_pdf, field, value_str_raw)

                if heuristic and is_fixed:
                    self.KB[label][field] = heuristic
                    log.info(f"Heurística {heuristic.type} (derivada) salva para '{field}'")
                elif not heuristic and is_fixed:
                     log.warning(f"Não foi possível derivar heurística para '{field}' (Valor: '{value_str_raw}')")
                
                value_str_clean = re.sub(r'\s*\n\s*', ', ', value_str_raw)
                final_data_clean[field] = value_str_clean

            return final_data_clean
            
        except json.JSONDecodeError as e:
            log.error(f"(LLM) Falha ao decodificar JSON do Bootstrap: {e}")
            log.error(f"Resposta recebida: {response_str}")
            return {field: None for field in extraction_schema.keys()}
        except Exception as e:
            log.error(f"(LLM) Erro inesperado no Bootstrap: {e}")
            return {field: None for field in extraction_schema.keys()}

    def _extract_variable_template(self, extraction_schema: Dict[str, str], parsed_pdf: Dict) -> Dict[str, Any]:
        """
        Extrai dados de um template variável (modo rápido).
        """
        log.info(f"(LLM) Template variável: Chamando LLM com prompt rápido.")
        self.stats['llm_fallbacks'] += 1 
        
        fields_json_str = json.dumps(extraction_schema, indent=2, ensure_ascii=False)
        text = parsed_pdf["full_text"][:8000] # Otimização: Trunca o texto
        
        prompt = f"""Extraia os campos do texto. Responda APENAS com JSON. Use 'null' se não encontrar.

        Campos:
        {fields_json_str}

        Texto:
        ---
        {text}
        ---
        """
        
        # Chama o _call_llm (que usará o gpt-5-mini padrão)
        response_str = self._call_llm(prompt, json_mode=True)
        
        try:
            extracted_data_raw = json.loads(response_str)
            
            extracted_data_clean = {}
            for field, value in extracted_data_raw.items():
                if field not in extraction_schema:
                    continue
                value_str = str(value).strip() if value else None
                if value_str:
                    value_str = re.sub(r'\s*\n\s*', ', ', value_str)
                extracted_data_clean[field] = value_str
            
            # Garante que todos os campos do schema estejam presentes
            for field in extraction_schema.keys():
                if field not in extracted_data_clean:
                    extracted_data_clean[field] = None
                    
            return extracted_data_clean
            
        except Exception as e:
            log.error(f"  ❌(LLM) Falha ao extrair template variável: {e}")
            return {field: None for field in extraction_schema.keys()}

    # --- NOVO: Fallback de LLM para um único documento ---
    def _single_doc_llm_fallback(self, req: ExtractRequest):
        """
        Chama o LLM para campos específicos que falharam nas heurísticas
        para um ÚNICO documento.
        """
        if not req._failed_fields:
            return

        log.info(f"  (LLM Fallback) Processando {len(req._failed_fields)} campos falhados para o label '{req.label}'...")
        self.stats['llm_fallbacks'] += 1
        
        # 1. Cria o schema apenas com os campos que falharam
        failed_schema = {
            field: req.extraction_schema[field] 
            for field in req._failed_fields
        }
        fields_json_str = json.dumps(failed_schema, indent=2, ensure_ascii=False)
        text = req._parsed_pdf["full_text"]

        # 2. Cria o prompt
        prompt = f"""Extraia APENAS os seguintes campos do texto. Responda APENAS com JSON. Use 'null' se não encontrar.

        Campos para extrair:
        {fields_json_str}

        Texto do Documento (com Layout):
        ---
        {text}
        ---
        """
        
        # 3. Chama o LLM
        response_str = self._call_llm(prompt, json_mode=True)
        
        if not response_str:
            log.error("  (LLM Fallback) Falha. Nenhuma resposta do LLM.")
            # Os campos já estão como None em req._result, então apenas retornamos
            return

        # 4. Processa a resposta
        try:
            batch_results = json.loads(response_str)
            
            for field in req._failed_fields:
                value = batch_results.get(field)
                
                if value and str(value).lower() != 'null':
                    value_str = str(value)
                    log.info(f"  (LLM Fallback) SUCESSO para '{field}'")
                    # Salva o resultado diretamente no objeto da requisição
                    req._result[field] = value_str
                    
                    # Tenta aprender a heurística e salvar no KB
                    new_heuristic = self._derive_heuristic_for_value(
                        req._parsed_pdf, field, value_str
                    )
                    if new_heuristic:
                        self.KB[req.label][field] = new_heuristic
                        log.info(f"  AUTOCORREÇÃO: Nova heurística '{new_heuristic.type}' salva para '{field}'")
                else:
                    log.warning(f"  (LLM Fallback) FALHA para '{field}' (retornou null)")
                    req._result[field] = None # Garante que seja None

        except json.JSONDecodeError as e:
            log.error(f"  (LLM Fallback) Falha ao decodificar JSON: {e}")
            log.error(f"  Resposta recebida: {response_str}")
        except Exception as e:
            log.error(f"  (LLM Fallback) Erro inesperado: {e}")

    # ==================== EXTRAÇÃO PRINCIPAL ====================
    
    def print_stats(self):
        print("\n" + "="*70)
        print("ESTATÍSTICAS DO EXTRATOR")
        print("="*70)
        print(f"Cache hits:           {self.stats['cache_hits']}")
        print(f"Extração por âncora:  {self.stats['anchor_extractions']}")
        print(f"Extração por zona:    {self.stats['zone_extractions']}")
        print(f"LLM Bootstraps (1ª vez): {self.stats['llm_bootstraps']}")
        print(f"LLM Fallbacks (indiv.): {self.stats['llm_fallbacks']}") # Atualizado
        print(f"Custo total:          ${self.stats['total_cost']:.6f}")
        print(f"Labels aprendidos:    {len(self.KB)}")
        
        total_heuristics = sum(len(fields) for fields in self.KB.values())
        print(f"Heurísticas salvas:   {total_heuristics}")
        print("="*70 + "\n")
    
    def export_kb(self, filepath: str):
        export_data = {
            'kb': {},
            'metadata': self.label_metadata
        }
        
        for label, fields in self.KB.items():
            export_data['kb'][label] = {}
            for field, heuristic in fields.items():
                export_data['kb'][label][field] = {
                    'type': heuristic.type,
                    'value': heuristic.value,
                    'confidence': heuristic.confidence,
                    'metadata': heuristic.metadata
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        log.info(f"Knowledge base exportado para: {filepath}")
    
    def import_kb(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.label_metadata = data.get('metadata', {})
            kb_data = data.get('kb', data)
            
            for label, fields in kb_data.items():
                self.KB[label] = {}
                for field, heur_data in fields.items():
                    if heur_data['type'] == "TABLE":
                        continue
                    self.KB[label][field] = Heuristic(
                        type=heur_data['type'],
                        value=heur_data['value'],
                        confidence=heur_data.get('confidence', 1.0),
                        metadata=heur_data.get('metadata', {})
                    )
            
            log.info(f"Knowledge base importado de: {filepath} (Labels: {len(self.KB)}, Metadados: {len(self.label_metadata)})")
            
        except Exception as e:
            log.error(f"Erro ao importar KB: {e}")

    # NOVO: Método para extração "burra" (sem label)
    def extract_unlabeled(self, extraction_schema: Dict[str, str], 
                          pdf_path: str) -> Dict[str, Any]:
        """
        Extrai dados de um PDF desconhecido, sem usar/salvar heurísticas.
        Sempre usa o LLM no modo "rápido" (template variável).
        """
        print(f"\n{'='*70}")
        print(f"Requisição (Sem Label) | {len(extraction_schema)} campos")
        print(f"PDF: {pdf_path}")
        print(f"{'='*70}")

        pdf_hash = self._get_pdf_hash(pdf_path)
        if pdf_hash and pdf_hash in self.pdf_cache:
            log.info(f"CACHE HIT (Resultado): PDF já processado")
            self.stats['cache_hits'] += 1
            cached_data = self.pdf_cache[pdf_hash]
            cleaned_cache = {}
            for field, value in cached_data.items():
                if isinstance(value, str):
                    cleaned_cache[field] = re.sub(r'\s*\n\s*', ', ', value)
                else:
                    cleaned_cache[field] = value
            return {field: cleaned_cache.get(field) for field in extraction_schema.keys()}
        
        parsed_pdf = None
        doc_to_close = None
        try:
            if pdf_hash and pdf_hash in self.parsed_pdf_cache:
                log.info(f"CACHE HIT (Parse): Usando PDF pré-parseado.")
                parsed_pdf = self.parsed_pdf_cache[pdf_hash]
                doc_to_close = fitz.open(pdf_path) 
                parsed_pdf["doc"] = doc_to_close
                parsed_pdf["page"] = doc_to_close[0]
            else:
                log.info(f"CACHE MISS (Parse): Parseando PDF...")
                parsed_pdf = self._parse_pdf(pdf_path)
                if parsed_pdf:
                    doc_to_close = parsed_pdf.get("doc")
                    cached_parse_data = parsed_pdf.copy()
                    cached_parse_data["doc"] = None 
                    cached_parse_data["page"] = None 
                    self.parsed_pdf_cache[pdf_hash] = cached_parse_data
            
            if not parsed_pdf:
                log.error(f"Falha ao parsear PDF")
                return {field: None for field in extraction_schema.keys()}

            log.info(">> Estratégia: Template Desconhecido (Chamada LLM individual rápida)...")
            extracted_data_clean = self._extract_variable_template(
                extraction_schema, parsed_pdf
            )
            
            if pdf_hash:
                raw_extracted_data = {f: (v.replace(', ', '\n') if isinstance(v, str) else None) for f, v in extracted_data_clean.items()}
                self.pdf_cache[pdf_hash] = raw_extracted_data

            return extracted_data_clean
            
        finally:
            if doc_to_close:
                doc_to_close.close()

    # --- MÉTODO PRINCIPAL DE LOTE (MODIFICADO) ---
    def extract_batch(self, requests: List[ExtractRequest]) -> Generator[Dict[str, Any], None, None]:
        """
        Processa uma lista de pedidos de extração de forma sequencial.
        Retorna (yield) o resultado de cada documento assim que ele é concluído.
        """
        
        log.info(f"Iniciando extração sequencial para {len(requests)} documentos...")
        
        # --- FASE 1: LOOP PRINCIPAL (SEQUENCIAL) ---
        for i, req in enumerate(requests):
            log.info(f"\n[Doc {i+1}/{len(requests)}] Processando: {req.pdf_path} (Label: '{req.label}')")
            
            req._pdf_hash = self._get_pdf_hash(req.pdf_path)
            if req._pdf_hash and req._pdf_hash in self.pdf_cache:
                log.info(f"CACHE HIT (Resultado): PDF já processado")
                self.stats['cache_hits'] += 1
                req._result = self.pdf_cache[req._pdf_hash]
                # Pula para a FASE 3 (Yield)
            
            else:
                # --- PARSE ---
                if req._pdf_hash and req._pdf_hash in self.parsed_pdf_cache:
                    log.info(f"CACHE HIT (Parse): Usando PDF pré-parseado.")
                    req._parsed_pdf = self.parsed_pdf_cache[req._pdf_hash]
                    doc = fitz.open(req.pdf_path)
                    req._parsed_pdf["doc"] = doc
                    req._parsed_pdf["page"] = doc[0]
                else:
                    log.info(f"CACHE MISS (Parse): Parseando PDF...")
                    req._parsed_pdf = self._parse_pdf(req.pdf_path)
                    if req._parsed_pdf:
                        cached_parse_data = req._parsed_pdf.copy()
                        cached_parse_data["doc"] = None 
                        cached_parse_data["page"] = None 
                        self.parsed_pdf_cache[req._pdf_hash] = cached_parse_data

                if not req._parsed_pdf:
                    log.error(f"Falha ao parsear PDF, pulando para FASE 3.")
                    req._result = {field: None for field in req.extraction_schema.keys()}
                    # Pula para a FASE 3 (Yield)

                # --- LÓGICA DE EXTRAÇÃO (Se o parse funcionou) ---
                elif req.label not in self.KB:
                    log.info(f"Label novo: '{req.label}'. Iniciando Bootstrap (chamada individual)...")
                    bootstrap_data_clean = self._bootstrap_new_label_with_llm(
                        req.label, req.extraction_schema, req._parsed_pdf
                    )
                    # Converte para o formato raw (com \n) para o cache
                    req._result = {f: (v.replace(', ', '\n') if isinstance(v, str) else None) for f, v in bootstrap_data_clean.items()}
                
                else:
                    is_fixed = self.label_metadata.get(req.label, {}).get('template_fixo', True)
                    
                    if not is_fixed:
                        log.info(">> Estratégia: Template Variável (Chamada LLM individual)...")
                        variable_data_clean = self._extract_variable_template(
                            req.extraction_schema, req._parsed_pdf
                        )
                        req._result = {f: (v.replace(', ', '\n') if isinstance(v, str) else None) for f, v in variable_data_clean.items()}
                    
                    else:
                        log.info(">> Estratégia: Template Fixo (Apenas Heurísticas)...")
                        for field, description in req.extraction_schema.items():
                            value = None
                            found = False
                            is_confirmed_empty = False 
                            
                            if field in self.KB[req.label]:
                                heuristic = self.KB[req.label][field]
                                value, found = self._apply_heuristic(heuristic, req._parsed_pdf)
                                
                                if found and value is None and heuristic.type == "ANCHOR_EMPTY":
                                    is_confirmed_empty = True
                            
                            if found and value is not None:
                                # log.info(f"Heurística '{heuristic.type}' encontrou valor para '{field}'.")
                                req._result[field] = value
                            
                            elif is_confirmed_empty:
                                log.info(f"Heurística 'ANCHOR_EMPTY' confirmou Nulo para '{field}'. (Sem LLM)")
                                req._result[field] = None
                            
                            else:
                                if not (field in self.KB[req.label]):
                                     log.warning(f"Nenhuma heurística salva para '{field}'. Adicionando à fila de LLM.")
                                else:
                                     log.warning(f"Heurística '{self.KB[req.label][field].type}' falhou em encontrar valor para '{field}'. Adicionando à fila de LLM.")
                                
                                req._failed_fields.append(field)
                                # Não adiciona à fila de lote, apenas marca como falha

                        # --- FASE 2: FALLBACK LLM (INDIVIDUAL) ---
                        # Esta chamada agora acontece DENTRO do loop de cada documento
                        if req._failed_fields:
                            log.info(f"[Doc {i+1}] {len(req._failed_fields)} falhas. Chamando LLM fallback individual...")
                            self._single_doc_llm_fallback(req) # Chama o novo método
                        else:
                            log.info(f"[Doc {i+1}] Sucesso total com heurísticas.")

            # --- FASE 3: FINALIZAR, SALVAR CACHE E RETORNAR (YIELD) ---
            final_data_raw = {}
            for field in req.extraction_schema.keys():
                final_data_raw[field] = req._result.get(field)

            # Salva no cache de resultado (se já não estiver lá)
            if req._pdf_hash and req._pdf_hash not in self.pdf_cache:
                self.pdf_cache[req._pdf_hash] = final_data_raw.copy()

            # Limpa os dados (formato \n -> ,) para o retorno
            cleaned_data = {}
            for field, value in final_data_raw.items():
                if isinstance(value, str):
                    cleaned_data[field] = re.sub(r'\s*\n\s*', ', ', value)
                else:
                    cleaned_data[field] = value
            
            # Fecha o documento PDF se ele foi aberto
            if req._parsed_pdf and req._parsed_pdf.get("doc"):
                req._parsed_pdf["doc"].close()
                req._parsed_pdf["doc"] = None
                req._parsed_pdf["page"] = None

            log.info(f"[Doc {i+1}/{len(requests)}] Documento finalizado e retornado.")
            yield cleaned_data

        log.info("Processamento em lote (sequencial) concluído.")
        # Fim da função (generators não precisam de return)


    # ATUALIZADO: Método `extract` agora consome o generator
    def extract(self, label: str, extraction_schema: Dict[str, str], 
                pdf_path: str) -> Dict[str, Any]:
        """
        Ponto de entrada para um único ficheiro.
        Envolve o novo método de extração em lote para reutilizar a lógica.
        """
        print(f"\n{'='*70}")
        print(f"Requisição: {label} | {len(extraction_schema)} campos")
        print(f"PDF: {pdf_path}")
        print(f"{'='*70}")
        
        single_request = ExtractRequest(
            label=label,
            extraction_schema=extraction_schema,
            pdf_path=pdf_path
        )
        
        # Converte o generator de volta para uma lista e pega o primeiro item
        batch_results = list(self.extract_batch([single_request]))
        
        return batch_results[0]