import fitz  # PyMuPDF
import re
import json
import logging
from openai import OpenAI
from typing import Optional, Any, List, Dict, Tuple, Generator 
from dataclasses import dataclass, field as dataclass_field
import hashlib
from collections import defaultdict

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

LLM_MODEL_NAME = "gpt-5-mini"


@dataclass
class Heuristic:
    """Representa uma heurística aprendida"""
    type: str  # ANCHOR, ZONE, ANCHOR_EMPTY
    value: Any
    confidence: float = 1.0
    metadata: Dict = dataclass_field(default_factory=dict)


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
    Extrator inteligente otimizado para <10s por documento.
    """
    
    def __init__(self, api_key: str):
        self.KB: Dict[str, Dict[str, Heuristic]] = {}
        self.label_metadata: Dict[str, Dict] = {} 
        
        self.pdf_cache: Dict[str, Dict] = {}
        self.parsed_pdf_cache: Dict[str, Dict] = {}
        self.llm_client = OpenAI(api_key=api_key)
        
        self.INPUT_COST = 0.150 / 1_000_000
        self.OUTPUT_COST = 0.600 / 1_000_000
        
        self.stats = {
            'cache_hits': 0,
            'anchor_extractions': 0,
            'zone_extractions': 0,
            'llm_bootstraps': 0,
            'llm_fallbacks': 0,
            'total_cost': 0.0
        }
        
        log.info(f"SmartExtractor Otimizado inicializado com {LLM_MODEL_NAME}.")

    # ==================== HELPERS ====================
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as f:
                sha256_hash = hashlib.sha256()
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
                return sha256_hash.hexdigest()
        except Exception as e:
            log.warning(f"Não foi possível gerar hash para {pdf_path}: {e}")
            return ""

    def _build_text_index(self, norm_words: List[Tuple[str, fitz.Rect]]) -> Dict[str, List[int]]:
        """
        OTIMIZAÇÃO: Cria índice invertido para busca O(1)
        """
        index = defaultdict(list)
        for i, (text, rect) in enumerate(norm_words):
            if text:
                index[text].append(i)
        return dict(index)

    def _parse_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        doc = None
        try:
            doc = fitz.open(pdf_path)
            if not doc or len(doc) == 0:
                log.warning(f"Documento PDF vazio ou inválido: {pdf_path}")
                return None
            
            page = doc[0]

            try:
                layout_text = page.get_text("text", sort=True)
            except Exception as sort_error:
                log.warning(f"Extração (modo 'sort=True') falhou: {sort_error}. Voltando para o modo padrão.")
                layout_text = page.get_text("text", sort=False) 

            words_data = page.get_text("words")
            if not words_data:
                log.error(f"PDF não contém texto legível (sem 'words'): {pdf_path}")
                return None

            # Pré-calcular palavras normalizadas
            norm_words = []
            for w in words_data:
                norm_text = self._normalize_text(w[4])
                if norm_text:
                    norm_words.append((norm_text, fitz.Rect(w[0:4])))
            
            # Criar índice invertido
            text_index = self._build_text_index(norm_words)
            
            # Truncar texto inteligentemente
            # Campos importantes raramente estão no final
            words = layout_text.split()
            if len(words) > 4000:
                layout_text = " ".join(words[:4000]) + "\n[...texto truncado...]"
            
            return {
                "doc": doc,
                "page": page,
                "words": words_data,
                "norm_words": norm_words,
                "text_index": text_index,
                "full_text": layout_text,
                "clean_text": layout_text
            }
            
        except Exception as e:
            log.exception(f"Erro ao parsear PDF: {pdf_path}, ERRO: {e}")
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

    def _search_for_normalized(self, page: fitz.Page, needle: str, norm_words: List[Tuple[str, fitz.Rect]], text_index: Dict[str, List[int]]) -> List[fitz.Rect]:
        """
        OTIMIZADO: Usa índice invertido para busca rápida
        """
        if not needle:
            return []
        
        needle_norm = self._normalize_text(needle)
        if not needle_norm:
            return []
        
        # Busca O(1) no índice
        if needle_norm in text_index:
            i = text_index[needle_norm][0]  # Pega primeira ocorrência
            return [norm_words[i][1]]
        
        # Fallback: busca multi-palavra
        for i in range(len(norm_words)):
            if needle_norm.startswith(norm_words[i][0]):
                current_text = norm_words[i][0]
                current_rect = fitz.Rect(norm_words[i][1])
                
                for j in range(i + 1, min(i + 5, len(norm_words))):
                    if norm_words[j][1].y0 > current_rect.y1 + 5:
                        break 
                    current_text += norm_words[j][0]
                    current_rect.include_rect(norm_words[j][1])
                    if current_text == needle_norm:
                        return [current_rect]
                    if not needle_norm.startswith(current_text):
                        break 
        
        return []

    # ==================== APLICADORES ====================
    
    def _apply_anchor_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Optional[str]:
        try:
            page = parsed_pdf["page"]
            anchor_text = heuristic.metadata.get("anchor_text")
            direction = heuristic.metadata.get("direction", "right")
            layout = heuristic.metadata.get("layout", "column") 
            multi_line = heuristic.metadata.get("multi_line", False)
            regex = heuristic.metadata.get("regex")
            
            if not anchor_text: return None
            
            norm_words_data = parsed_pdf.get("norm_words")
            text_index = parsed_pdf.get("text_index", {})
            anchor_rects = self._search_for_normalized(page, anchor_text, norm_words_data, text_index)
            if not anchor_rects:
                return None
            
            anchor_rect = anchor_rects[0]
            page_rect = page.rect
            
            search_rect = None
            
            if direction == "right":
                search_rect = fitz.Rect(
                    anchor_rect.x1 + 2, 
                    anchor_rect.y0 - 2,
                    page_rect.width - 10, 
                    anchor_rect.y1 + 2
                )
            elif direction == "left":
                search_rect = fitz.Rect(
                    10, 
                    anchor_rect.y0 - 2,
                    anchor_rect.x0 - 2, 
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
                else: # layout "row"
                    search_rect = fitz.Rect(
                        5, 
                        anchor_rect.y1 + 2,
                        page_rect.width - 10, 
                        height
                    )
            elif direction == "above":
                if multi_line:
                    height = anchor_rect.y0 - 70
                else:
                    height = anchor_rect.y0 - 20
                    
                if layout == "column":
                     search_rect = fitz.Rect(
                        anchor_rect.x0 - 10, 
                        height,
                        anchor_rect.x1 + 300, 
                        anchor_rect.y0 - 2
                    )
                else: # layout "row"
                    search_rect = fitz.Rect(
                        5, 
                        height,
                        page_rect.width - 10, 
                        anchor_rect.y0 - 2
                    )
            else:
                return None
            
            value_text = page.get_text("text", clip=search_rect)
            if not value_text: return None
            
            value_text = value_text.strip().replace(anchor_text, "").strip()
            
            if regex:
                # Se houver regex, aplicamos em todo o bloco
                match = re.search(regex, value_text, re.DOTALL) 
                value_text = match.group(0) if match else None
            else:
                # Lógica original de linha única / multi-linha
                lines = [l.strip() for l in value_text.split('\n') if l.strip()]
                if not lines:
                    value_text = None
                elif multi_line:
                    value_text = "\n".join(lines)
                else:
                    # Se for 'above' ou 'left', pegamos a última linha, não a primeira
                    if direction in ("above", "left"):
                        value_text = lines[-1]
                    else:
                        value_text = lines[0]
            
            if value_text:
                preview = value_text.replace('\n', ' ')
                preview = preview[:50] + "..." if len(preview) > 50 else preview
                log.info(f"Âncora: '{anchor_text}' ({direction}) → '{preview}'")
                return value_text
            
            return None
        except Exception as e:
            log.warning(f"Erro ao aplicar heurística de âncora: {e}")
            return None
    
    def _apply_zone_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Optional[str]:
        try:
            page = parsed_pdf["page"]
            zone_coords = heuristic.value
            regex = heuristic.metadata.get("regex")
            
            if not zone_coords or len(zone_coords) != 4: return None
            
            zone_rect = fitz.Rect(zone_coords)
            value_text = page.get_text("text", clip=zone_rect)
            
            if value_text and value_text.strip():
                value_text = value_text.strip()
                
                if regex:
                    match = re.search(regex, value_text, re.DOTALL)
                    value_text = match.group(0) if match else None

                if value_text:
                    log.info(f"Zona: {zone_coords} (Regex: {bool(regex)}) → '{value_text[:50]}...'")
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
            log.info(f"Aplicando heurística ANCHOR_EMPTY")
            anchor_text = heuristic.metadata.get("anchor_text")
            if not anchor_text: 
                return None, False 
            
            value = self._apply_anchor_heuristic(heuristic, parsed_pdf) 
            
            if value:
                log.info(f"AUTOCORREÇÃO: ANCHOR_EMPTY encontrou valor! '{value}'")
                return value, True 
            
            log.info(f"ANCHOR_EMPTY falhou. Acionando LLM.")
            return None, False
        else:
            return None, False

        if value is not None:
            return value, True
        else:
            return None, False

    # ==================== NLP FALLBACK ====================
    
    def _learn_from_anchor(self, parsed_pdf: Dict, field: str, description: str) -> Tuple[Optional[str], Optional[Heuristic], bool]:
        page = parsed_pdf["page"]
        page_rect = page.rect
        norm_words_data = parsed_pdf.get("norm_words")
        text_index = parsed_pdf.get("text_index", {})
        
        anchor_candidates = list(set([
            field, field.replace("_", " "), field.replace("_", " ").title(), 
            field.upper(), description, description.upper()
        ]))

        log.info(f"  🔍(NLP) Procurando âncoras: {anchor_candidates[:3]}...")
        
        for anchor in anchor_candidates:
            if not anchor: continue
            
            rects = self._search_for_normalized(page, anchor, norm_words_data, text_index)
            if not rects: continue
            
            anchor_rect = rects[0]
            
            # Direita
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
            
            # Abaixo
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
            
            log.info(f"(NLP) ÂNCORA '{anchor}' encontrada mas valor VAZIO.")
            heuristic = Heuristic(
                type="ANCHOR_EMPTY", value=None, confidence=0.8,
                metadata={"anchor_text": anchor, "direction": "right"} 
            )
            return None, heuristic, True

        return None, None, False

    # ==================== LLM ====================
    
    def _call_llm(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        try:
            log.info(f"Chamando LLM... (JSON Mode: {json_mode})")
            
            model_params = {
                "model": LLM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Você é um assistente especialista em extração de dados de documentos. Siga as instruções de formato da resposta com precisão."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1.0
            }
            if json_mode:
                model_params["response_format"] = {"type": "json_object"}

            response = self.llm_client.chat.completions.create(**model_params)
            
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
    
    def _guess_regex_for_value(self, value: str) -> Optional[str]:
        """Tenta adivinhar um regex comum para o valor extraído."""
        if not value:
            return None
        
        # CPF
        if re.fullmatch(r'\d{3}\.\d{3}\.\d{3}-\d{2}', value):
            return r'\d{3}\.\d{3}\.\d{3}-\d{2}'
        # CNPJ
        if re.fullmatch(r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}', value):
            return r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}'
        # Data (DD/MM/AAAA)
        if re.fullmatch(r'\d{2}/\d{2}/\d{4}', value):
            return r'\d{2}/\d{2}/\d{4}'
        # CEP
        if re.fullmatch(r'\d{5}-\d{3}', value):
            return r'\d{5}-\d{3}'
        # Apenas números (Inscrição, etc)
        if re.fullmatch(r'\d+', value):
            return r'\d+'
        # Valor monetário (BRL)
        if re.fullmatch(r'R\$\s*[\d\.,]+', value) or re.fullmatch(r'[\d\.,]+', value):
             # Regex mais permissivo para capturar valores
             if any(c in value for c in '.,'):
                return r'[\d\.,]+'

        return None

    def _derive_heuristic_for_value(self, parsed_pdf: Dict, field: str, value: str) -> Optional[Heuristic]:
        page = parsed_pdf["page"]
        norm_words_data = parsed_pdf.get("norm_words")
        text_index = parsed_pdf.get("text_index", {})
        if not value: return None
        
        clean_value = value.strip().replace(',', ' ').replace('\n', ' ')
        value_parts = clean_value.split()
        if len(value_parts) == 0:
            return None
        
        # Tenta encontrar o melhor regex para o valor
        regex = self._guess_regex_for_value(value.strip().split('\n')[0])
        log.info(f"Regex aprendido para '{field}': {regex}")
        
        value_for_search = " ".join(value_parts[0:3])
        value_rects = self._search_for_normalized(page, value_for_search, norm_words_data, text_index)
        
        if not value_rects:
            value_for_search = " ".join(value_parts[0:1])
            value_rects = self._search_for_normalized(page, value_for_search, norm_words_data, text_index)
            
            if not value_rects:
                log.warning(f"Não foi possível encontrar '{value_for_search}' para derivar heurística.")
                return None
        
        rect = value_rects[0]
        metadata = {}
        if regex:
            metadata["regex"] = regex
        
        # Tenta âncora acima (Valor está "below" da âncora)
        anchor_rect_above = fitz.Rect(rect.x0 - 50, max(0, rect.y0 - 50), rect.x1 + 50, rect.y0 - 2)
        anchor_text_above = page.get_text("text", clip=anchor_rect_above).strip()
        if anchor_text_above:
            anchor = anchor_text_above.split('\n')[-1].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"Heurística ÂNCORA (acima): '{anchor}' → '{value[:20]}...'")
                meta = metadata.copy()
                meta.update({"anchor_text": anchor, "direction": "below"})
                return Heuristic(type="ANCHOR", value=None, confidence=0.9, metadata=meta)

        # Tenta âncora à esquerda (Valor está "right" da âncora)
        anchor_rect_left = fitz.Rect(max(0, rect.x0 - 300), rect.y0 - 5, rect.x0 - 2, rect.y1 + 5)
        anchor_text_left = page.get_text("text", clip=anchor_rect_left).strip()
        if anchor_text_left:
            anchor = anchor_text_left.split('\n')[-1].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"Heurística ÂNCORA (esquerda): '{anchor}' → '{value[:20]}...'")
                meta = metadata.copy()
                meta.update({"anchor_text": anchor, "direction": "right"})
                return Heuristic(type="ANCHOR", value=None, confidence=0.9, metadata=meta)
        
        # Tenta âncora abaixo (Valor está "above" da âncora)
        anchor_rect_below = fitz.Rect(rect.x0 - 50, rect.y1 + 2, rect.x1 + 50, rect.y1 + 50)
        anchor_text_below = page.get_text("text", clip=anchor_rect_below).strip()
        if anchor_text_below:
            anchor = anchor_text_below.split('\n')[0].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"Heurística ÂNCORA (abaixo): '{anchor}' → '{value[:20]}...'")
                meta = metadata.copy()
                meta.update({"anchor_text": anchor, "direction": "above"})
                return Heuristic(type="ANCHOR", value=None, confidence=0.8, metadata=meta)

        # Tenta âncora à direita (Valor está "left" da âncora)
        anchor_rect_right = fitz.Rect(rect.x1 + 2, rect.y0 - 5, rect.x1 + 300, rect.y1 + 5)
        anchor_text_right = page.get_text("text", clip=anchor_rect_right).strip()
        if anchor_text_right:
            anchor = anchor_text_right.split('\n')[0].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"Heurística ÂNCORA (direita): '{anchor}' → '{value[:20]}...'")
                meta = metadata.copy()
                meta.update({"anchor_text": anchor, "direction": "left"})
                return Heuristic(type="ANCHOR", value=None, confidence=0.8, metadata=meta)

        # ZONA horizontal (Fallback)
        page_rect = page.rect
        y0 = max(0, rect.y0 - 5)
        y1 = min(page_rect.height - 2, rect.y1 + 5) 
        x0 = 5.0
        x1 = page_rect.width - 5.0

        if '\n' in value or len(clean_value) > 80:
            y1 = min(page_rect.height - 2, rect.y1 + 70) 
        
        zone_coords = [x0, y0, x1, y1]
        log.info(f"ZONA 'Horizontal Slice': {zone_coords}")
        
        return Heuristic(type="ZONE", value=zone_coords, confidence=0.7, metadata=metadata)

    def _bootstrap_new_label_with_llm(self, label: str, extraction_schema: Dict[str, str], parsed_pdf: Dict) -> Dict[str, Any]:
        log.info(f"(LLM) Bootstrap: Label '{label}'")
        self.stats['llm_bootstraps'] += 1
        
        # Prompt mais enxuto
        fields_list = "\n".join([f'"{k}": "{v}"' for k, v in extraction_schema.items()])
        text = parsed_pdf["full_text"]  # Já truncado no parse
        
        prompt = f"""
            Extraia dados do documento e retorne JSON com:
            1. "template_fixo": true (layout fixo como RG/OAB) ou false (variável como nota fiscal)
            2. "fields": objeto com os dados extraídos (use null se não encontrar)

            Campos:
            {fields_list}

            Documento:
            ---
            {text}
            ---
            JSON:
        """
        
        response_str = self._call_llm(prompt, json_mode=True)
        
        if not response_str:
            log.error("Bootstrap falhou")
            return {field: None for field in extraction_schema.keys()}
        
        try:
            response_json = json.loads(response_str)
            
            is_fixed = response_json.get('template_fixo', True)
            self.label_metadata[label] = {'template_fixo': is_fixed}
            log.info(f"Template: {'FIXO' if is_fixed else 'VARIÁVEL'}")

            final_data_clean = {} 
            self.KB[label] = {}
            
            fields_data = response_json.get("fields", {})
            for field in extraction_schema.keys():
                
                value_str_raw = str(fields_data.get(field)).strip() if fields_data.get(field) else None
                
                if not value_str_raw or value_str_raw.lower() == 'null':
                    log.info(f"Bootstrap: '{field}' → NÃO ENCONTRADO")
                    final_data_clean[field] = None
                    _, heuristic, found = self._learn_from_anchor(parsed_pdf, field, extraction_schema[field])
                    if found and heuristic and is_fixed:
                        self.KB[label][field] = heuristic
                    continue

                log.info(f"Bootstrap: '{field}' → '{value_str_raw[:50].replace(chr(10), ' ')}...'")
                
                heuristic = self._derive_heuristic_for_value(parsed_pdf, field, value_str_raw)

                if heuristic and is_fixed:
                    self.KB[label][field] = heuristic
                    log.info(f"Heurística {heuristic.type} salva")
                
                value_str_clean = re.sub(r'\s*\n\s*', ', ', value_str_raw)
                final_data_clean[field] = value_str_clean

            return final_data_clean
            
        except json.JSONDecodeError as e:
            log.error(f"Falha JSON: {e}")
            return {field: None for field in extraction_schema.keys()}
        except Exception as e:
            log.error(f"Erro Bootstrap: {e}")
            return {field: None for field in extraction_schema.keys()}

    def _extract_variable_template(self, extraction_schema: Dict[str, str], parsed_pdf: Dict) -> Dict[str, Any]:
        log.info(f"(LLM) Template variável")
        self.stats['llm_fallbacks'] += 1 

        fields_list = "\n".join([f'"{k}": "{v}"' for k, v in extraction_schema.items()])
        text = parsed_pdf["full_text"]  # Já truncado
        
        prompt = f"""
            Extraia os campos. Retorne apenas JSON (use null se não encontrar):

            {fields_list}

            Documento:
            ---
            {text}
            ---
            JSON:
        """
        
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
            
            for field in extraction_schema.keys():
                if field not in extracted_data_clean:
                    extracted_data_clean[field] = None
                    
            return extracted_data_clean
            
        except Exception as e:
            log.error(f"Falha template variável: {e}")
            return {field: None for field in extraction_schema.keys()}

    def _single_doc_llm_fallback(self, req: ExtractRequest):
        if not req._failed_fields:
            return

        log.info(f"  (LLM Fallback) {len(req._failed_fields)} campos")
        self.stats['llm_fallbacks'] += 1
        
        failed_schema = {
            field: req.extraction_schema[field] 
            for field in req._failed_fields
        }
        fields_list = "\n".join([f'"{k}": "{v}"' for k, v in failed_schema.items()])
        text = req._parsed_pdf["full_text"]

        prompt = f"""
            Extraia apenas estes campos (use null se não encontrar):

            {fields_list}

            Documento:
            ---
            {text}
            ---
            JSON:
        """

        response_str = self._call_llm(prompt, json_mode=True)

        if not response_str:
            log.error("  Fallback falhou")
            return

        try:
            batch_results = json.loads(response_str)
            
            for field in req._failed_fields:
                value = batch_results.get(field)
                
                if value and str(value).lower() != 'null':
                    value_str = str(value)
                    log.info(f"  Fallback SUCESSO: '{field}'")
                    req._result[field] = value_str
                    
                    new_heuristic = self._derive_heuristic_for_value(
                        req._parsed_pdf, field, value_str
                    )
                    if new_heuristic:
                        self.KB[req.label][field] = new_heuristic
                        log.info(f"  AUTOCORREÇÃO: '{new_heuristic.type}' salva")
                else:
                    log.warning(f"  Fallback FALHA: '{field}'")
                    req._result[field] = None

        except Exception as e:
            log.error(f"  Fallback erro: {e}")

    # ==================== EXTRAÇÃO PRINCIPAL ====================
    
    def print_stats(self):
        print("\n" + "="*70)
        print("ESTATÍSTICAS")
        print("="*70)
        print(f"Cache hits:          {self.stats['cache_hits']}")
        print(f"Âncoras:             {self.stats['anchor_extractions']}")
        print(f"Zonas:               {self.stats['zone_extractions']}")
        print(f"LLM Bootstraps:      {self.stats['llm_bootstraps']}")
        print(f"LLM Fallbacks:       {self.stats['llm_fallbacks']}")
        print(f"Custo total:         ${self.stats['total_cost']:.6f}")
        print(f"Labels aprendidos:   {len(self.KB)}")
        
        total_heuristics = sum(len(fields) for fields in self.KB.values())
        print(f"Heurísticas salvas:  {total_heuristics}")
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
        
        log.info(f"KB exportado: {filepath}")
    
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
            
            log.info(f"KB importado: {filepath} ({len(self.KB)} labels)")
            
        except Exception as e:
            log.error(f"Erro ao importar KB: {e}")

    def extract_unlabeled(self, extraction_schema: Dict[str, str], 
                          pdf_path: str) -> Dict[str, Any]:
        """
        Extrai dados sem usar/salvar heurísticas (modo rápido).
        """
        print(f"\n{'='*70}")
        print(f"Requisição (Sem Label) | {len(extraction_schema)} campos")
        print(f"PDF: {pdf_path}")
        print(f"{'='*70}")

        pdf_hash = self._get_pdf_hash(pdf_path)
        if pdf_hash and pdf_hash in self.pdf_cache:
            log.info(f"CACHE HIT")
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
                log.info(f"CACHE HIT (Parse)")
                parsed_pdf = self.parsed_pdf_cache[pdf_hash]
                doc_to_close = fitz.open(pdf_path) 
                parsed_pdf["doc"] = doc_to_close
                parsed_pdf["page"] = doc_to_close[0]
            else:
                log.info(f"Parseando PDF...")
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

            log.info(">> Template Desconhecido (LLM)")
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

    def extract_batch(self, requests: List[ExtractRequest]) -> Generator[Dict[str, Any], None, None]:
        """
        Processa documentos de forma sequencial.
        Retorna (yield) cada resultado assim que concluído.
        """
        
        log.info(f"Iniciando extração para {len(requests)} documentos...")
        
        for i, req in enumerate(requests):
            log.info(f"\n[Doc {i+1}/{len(requests)}] {req.pdf_path} (Label: '{req.label}')")
            
            req._pdf_hash = self._get_pdf_hash(req.pdf_path)
            if req._pdf_hash and req._pdf_hash in self.pdf_cache:
                log.info(f"CACHE HIT")
                self.stats['cache_hits'] += 1
                req._result = self.pdf_cache[req._pdf_hash]
            
            else:
                # Parse
                if req._pdf_hash and req._pdf_hash in self.parsed_pdf_cache:
                    log.info(f"CACHE HIT (Parse)")
                    req._parsed_pdf = self.parsed_pdf_cache[req._pdf_hash]
                    doc = fitz.open(req.pdf_path)
                    req._parsed_pdf["doc"] = doc
                    req._parsed_pdf["page"] = doc[0]
                else:
                    log.info(f"Parseando PDF...")
                    req._parsed_pdf = self._parse_pdf(req.pdf_path)
                    if req._parsed_pdf:
                        cached_parse_data = req._parsed_pdf.copy()
                        cached_parse_data["doc"] = None 
                        cached_parse_data["page"] = None 
                        self.parsed_pdf_cache[req._pdf_hash] = cached_parse_data

                if not req._parsed_pdf:
                    log.error(f"Falha no parse")
                    req._result = {field: None for field in req.extraction_schema.keys()}

                # Extração
                elif req.label not in self.KB:
                    log.info(f"Label novo: '{req.label}'. Bootstrap...")
                    bootstrap_data_clean = self._bootstrap_new_label_with_llm(
                        req.label, req.extraction_schema, req._parsed_pdf
                    )
                    req._result = {f: (v.replace(', ', '\n') if isinstance(v, str) else None) for f, v in bootstrap_data_clean.items()}
                
                else:
                    is_fixed = self.label_metadata.get(req.label, {}).get('template_fixo', True)
                    
                    if not is_fixed:
                        log.info(">> Template Variável (LLM)")
                        variable_data_clean = self._extract_variable_template(
                            req.extraction_schema, req._parsed_pdf
                        )
                        req._result = {f: (v.replace(', ', '\n') if isinstance(v, str) else None) for f, v in variable_data_clean.items()}
                    
                    else:
                        log.info(">> Template Fixo (Heurísticas)")
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
                                req._result[field] = value
                            
                            elif is_confirmed_empty:
                                log.info(f"ANCHOR_EMPTY confirmou Nulo: '{field}'")
                                req._result[field] = None
                            
                            else:
                                if not (field in self.KB[req.label]):
                                     log.warning(f"Sem heurística: '{field}' → LLM")
                                else:
                                     log.warning(f"Heurística falhou: '{field}' → LLM")
                                
                                req._failed_fields.append(field)

                        # Fallback LLM individual
                        if req._failed_fields:
                            log.info(f"[Doc {i+1}] {len(req._failed_fields)} falhas. LLM fallback...")
                            self._single_doc_llm_fallback(req)
                        else:
                            log.info(f"[Doc {i+1}] Sucesso total com heurísticas")

            # Finalizar
            final_data_raw = {}
            for field in req.extraction_schema.keys():
                final_data_raw[field] = req._result.get(field)

            if req._pdf_hash and req._pdf_hash not in self.pdf_cache:
                self.pdf_cache[req._pdf_hash] = final_data_raw.copy()

            # Limpa dados para retorno
            cleaned_data = {}
            for field, value in final_data_raw.items():
                if isinstance(value, str):
                    cleaned_data[field] = re.sub(r'\s*\n\s*', ', ', value)
                else:
                    cleaned_data[field] = value
            
            # Fecha PDF
            if req._parsed_pdf and req._parsed_pdf.get("doc"):
                req._parsed_pdf["doc"].close()
                req._parsed_pdf["doc"] = None
                req._parsed_pdf["page"] = None

            log.info(f"[Doc {i+1}/{len(requests)}] Concluído")
            yield cleaned_data

        log.info("Processamento em lote concluído")

    def extract(self, label: str, extraction_schema: Dict[str, str], 
                pdf_path: str) -> Dict[str, Any]:
        """
        Extrai dados de um único documento.
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
        
        batch_results = list(self.extract_batch([single_request]))
        
        return batch_results[0]