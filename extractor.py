import fitz  # PyMuPDF
import re
import json
import logging
from openai import OpenAI
from typing import Optional, Any, List, Dict, Tuple
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
    type: str  # TABLE, ANCHOR, ZONE, REGEX, ANCHOR_EMPTY
    value: Any
    confidence: float = 1.0
    metadata: Dict = dataclass_field(default_factory=dict)


# NOVO: Dataclass para gerir pedidos em lote
@dataclass
class ExtractRequest:
    """Representa um único pedido de extração num lote."""
    label: str
    extraction_schema: Dict[str, str]
    pdf_path: str
    
    # Campos de estado interno, não preencha
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
        
        self.pdf_cache: Dict[str, Dict] = {}
        self.llm_client = OpenAI(api_key=api_key)
        
        self.INPUT_COST = 0.150 / 1_000_000
        self.OUTPUT_COST = 0.600 / 1_000_000
        
        self.stats = {
            'cache_hits': 0,
            'table_extractions': 0,
            'anchor_extractions': 0,
            'zone_extractions': 0,
            'nlp_learns': 0,
            'llm_bootstraps': 0,
            'llm_fallbacks': 0,
            'total_cost': 0.0
        }
        
        log.info(f"SmartExtractor (Bootstrap LLM) inicializado com {LLM_MODEL_NAME}.")

    # ==================== HELPERS ====================
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as f:
                return hashlib.sha256(f.read(65536)).hexdigest()
        except:
            return ""
    
    # ATUALIZADO: _parse_pdf corrigido
    def _parse_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        doc = None
        try:
            doc = fitz.open(pdf_path)
            if not doc or len(doc) == 0:
                log.warning(f"  ⚠️  Documento PDF vazio ou inválido: {pdf_path}")
                return None
            
            page = doc[0]

            # --- DADOS PARA O LLM (com fallback) ---
            try:
                # 1. Tenta o modo "sort=True" (ideal para o LLM ver o layout)
                layout_text = page.get_text("text", sort=True)
                log.info("  ✓ Extração de texto (modo 'sort=True') bem-sucedida.")
            except Exception as sort_error:
                # 2. Se a ordenação falhar, volta para o modo padrão (texto empilhado)
                log.warning(f"  ⚠️  Extração (modo 'sort=True') falhou: {sort_error}. Voltando para o modo padrão (lógico).")
                layout_text = page.get_text("text", sort=False) # Modo padrão

            print("FULL TEXT (final usado para LLM):", layout_text)
            
            # --- DADOS PARA AS HEURÍSTICAS ---
            # Isto devolve a lista de palavras com coordenadas [x0, y0, x1, y1, "palavra", ...]
            words_data = page.get_text("words")
            if not words_data:
                log.error(f"❌ PDF não contém texto legível (sem 'words'): {pdf_path}")
                return None
            
            tables = []
            try:
                table_finder = page.find_tables()
                if table_finder:
                    for table in table_finder:
                        extracted = table.extract()
                        if extracted:
                            tables.append(extracted)
                    log.info(f"  📊 {len(tables)} tabela(s) encontrada(s)")
            except Exception as e:
                log.warning(f"  ⚠️  Erro ao extrair tabelas: {e}")
            
            return {
                "doc": doc,
                "page": page,
                "words": words_data,       # Para _search_for_normalized (lista de palavras)
                "full_text": layout_text,  # Para o LLM (string com layout)
                "clean_text": layout_text, # Para o LLM (string com layout)
                "tables": tables
            }
            
        except Exception as e:
            # Captura erros do fitz.open() ou outros erros fatais
            log.exception(f"❌ Erro detalhado ao parsear PDF: {pdf_path}")
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

    def _is_similar(self, s1: str, s2: str, threshold: float = 0.8) -> bool:
        n1 = self._normalize_text(s1)
        n2 = self._normalize_text(s2)
        if n1 == n2: return True
        if n1 in n2 or n2 in n1: return True
        if len(n1) > 3 and len(n2) > 3:
            max_len = max(len(n1), len(n2))
            min_len = min(len(n1), len(n2))
            if min_len / max_len >= threshold: return True
        return False

    # NOVO: Helper de busca que corrige o bug do 'ç'
    def _search_for_normalized(self, page: fitz.Page, needle: str, words_data: List[Tuple]) -> List[fitz.Rect]:
        """
        Substituto para page.search_for() que usa _normalize_text 
        para ignorar acentos e capitalização.
        """
        if not needle:
            return []
        
        # 1. Normaliza a 'needle' (o que procuramos), ex: "enderecoprofissional"
        needle_norm = self._normalize_text(needle)
        if not needle_norm:
            return []
        
        # 2. Pré-processa as palavras do PDF
        norm_words = []
        for w in words_data:
            norm_text = self._normalize_text(w[4])
            if norm_text:
                norm_words.append((norm_text, fitz.Rect(w[0:4])))
        
        # 3. Itera pelas palavras normalizadas do PDF
        for i in range(len(norm_words)):
            
            # 3a. Tenta correspondência de palavra única
            if norm_words[i][0] == needle_norm:
                return [norm_words[i][1]] # Encontrado

            # 3b. Tenta correspondência de múltiplas palavras (ex: "Endereço Profissional")
            if needle_norm.startswith(norm_words[i][0]):
                current_text = norm_words[i][0]
                current_rect = fitz.Rect(norm_words[i][1])
                
                # Procura até 5 palavras à frente
                for j in range(i + 1, min(i + 5, len(norm_words))):
                    # Verifica se as palavras estão próximas (na mesma linha)
                    if norm_words[j][1].y0 > current_rect.y1 + 5:
                        break # Palavra está muito abaixo
                        
                    current_text += norm_words[j][0]
                    current_rect.include_rect(norm_words[j][1])
                    
                    if current_text == needle_norm:
                        return [current_rect] # Encontrado
                    
                    if not needle_norm.startswith(current_text):
                        break # Caminho errado, pára de procurar
        
        return [] # Não encontrado

    # ==================== NÍVEL 1: APLICADORES ====================
    
    def _apply_table_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Optional[str]:
        try:
            header = heuristic.metadata.get("header")
            col_index = heuristic.metadata.get("col_index")
            row_index = heuristic.metadata.get("row_index", 1)
            
            if not header or col_index is None: return None
            
            tables = parsed_pdf.get("tables", [])
            for table in tables:
                if not table or len(table) == 0: continue
                headers = [str(h).strip() for h in table[0]]
                for i, h in enumerate(headers):
                    if self._is_similar(header, h):
                        if len(table) > row_index and len(table[row_index]) > i:
                            value = table[row_index][i]
                            if value:
                                log.info(f"    ✓ Tabela: '{header}' → '{value}'")
                                return str(value).strip()
            return None
        except Exception as e:
            log.warning(f"  ⚠️  Erro ao aplicar heurística de tabela: {e}")
            return None
    
    # ATUALIZADO: _apply_anchor_heuristic
    def _apply_anchor_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Optional[str]:
        try:
            page = parsed_pdf["page"]
            anchor_text = heuristic.metadata.get("anchor_text")
            direction = heuristic.metadata.get("direction", "right")
            layout = heuristic.metadata.get("layout", "column") 
            multi_line = heuristic.metadata.get("multi_line", False)
            regex = heuristic.metadata.get("regex")
            
            if not anchor_text: return None

            print('PAGINA:', page)
            
            # **** MUDANÇA AQUI: Usa a nova busca normalizada ****
            words_data = parsed_pdf.get("words")
            anchor_rects = self._search_for_normalized(page, anchor_text, words_data)
            if not anchor_rects:
                log.warning(f"  ⚠️  Não foi possível encontrar âncora (normalizada): '{anchor_text}'")
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
                    height = anchor_rect.y1 + 70 # Alto para endereços
                else:
                    height = anchor_rect.y1 + 20 # Curto para campos únicos
                
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
                log.info(f"    ✓ Âncora: '{anchor_text}' ({direction}/{layout}) → '{preview}'")
                return value_text
            
            return None
        except Exception as e:
            log.warning(f"  ⚠️  Erro ao aplicar heurística de âncora: {e}")
            return None
    
    def _apply_zone_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Optional[str]:
        # (Este método está correto)
        try:
            page = parsed_pdf["page"]
            zone_coords = heuristic.value
            if not zone_coords or len(zone_coords) != 4: return None
            
            zone_rect = fitz.Rect(zone_coords)
            value_text = page.get_text("text", clip=zone_rect)
            
            if value_text and value_text.strip():
                value_text = value_text.strip()
                log.info(f"    ✓ Zona: {zone_coords} → '{value_text}'")
                return value_text
            
            return None
        except Exception as e:
            log.warning(f"  ⚠️  Erro ao aplicar heurística de zona: {e}")
            return None

    def _apply_heuristic(self, heuristic: Heuristic, parsed_pdf: Dict) -> Tuple[Optional[str], bool]:
        # (Este método está correto)
        value = None
        
        if heuristic.type == "TABLE":
            self.stats['table_extractions'] += 1
            value = self._apply_table_heuristic(heuristic, parsed_pdf)
        elif heuristic.type == "ANCHOR":
            self.stats['anchor_extractions'] += 1
            value = self._apply_anchor_heuristic(heuristic, parsed_pdf)
        elif heuristic.type == "ZONE":
            self.stats['zone_extractions'] += 1
            value = self._apply_zone_heuristic(heuristic, parsed_pdf)
        elif heuristic.type == "ANCHOR_EMPTY":
            log.info(f"    ✓ Aplicando heurística ANCHOR_EMPTY (esperado Nulo)")
            anchor_text = heuristic.metadata.get("anchor_text")
            if not anchor_text: return None, False 
            value = self._apply_anchor_heuristic(heuristic, parsed_pdf)
            if value:
                log.info(f"    ✨ AUTOCORREÇÃO: ANCHOR_EMPTY encontrou um valor! '{value}'")
                return value, True 
            log.info(f"    ✓ ANCHOR_EMPTY confirmado como Nulo.")
            return None, True 
        else:
            return None, False
            
        if value is not None:
            return value, True
        else:
            return None, False

    # ==================== NÍVEL 2: APRENDIZADO NLP (FALLBACK) ====================
    
    def _learn_from_table(self, parsed_pdf: Dict, field: str, description: str) -> Tuple[Optional[str], Optional[Heuristic]]:
        # (Este método está correto)
        tables = parsed_pdf.get("tables", [])
        if not tables: return None, None
        
        log.info(f"  🔍(NLP) Procurando '{field}' em {len(tables)} tabela(s)...")
        field_variations = [field, field.replace("_", " "), field.replace("_", " ").title(), field.upper(), field.title()]
        
        for table_idx, table in enumerate(tables):
            if not table or len(table) == 0: continue
            headers = [str(h).strip() for h in table[0]]
            for col_idx, header in enumerate(headers):
                for variation in field_variations:
                    if self._is_similar(variation, header):
                        if len(table) > 1 and len(table[1]) > col_idx:
                            value = table[1][col_idx]
                            value_str = str(value).strip() if value else ""
                            heuristic = Heuristic(
                                type="TABLE", value=None, confidence=1.0,
                                metadata={"header": header, "col_index": col_idx, "row_index": 1, "table_index": table_idx}
                            )
                            log.info(f"  ✅(NLP) TABELA: '{header}' → '{value_str}'")
                            return value_str if value_str else None, heuristic
        return None, None
    
    # ATUALIZADO: _learn_from_anchor
    def _learn_from_anchor(self, parsed_pdf: Dict, field: str, description: str) -> Tuple[Optional[str], Optional[Heuristic], bool]:
        page = parsed_pdf["page"]
        page_rect = page.rect
        words_data = parsed_pdf.get("words")
        
        anchor_candidates = list(set([
            field, field.replace("_", " "), field.replace("_", " ").title(), 
            field.upper(), description, description.upper()
        ]))

        log.info(f"  🔍(NLP) Procurando âncoras: {anchor_candidates[:3]}...")
        
        for anchor in anchor_candidates:
            if not anchor: continue
            
            # **** MUDANÇA AQUI: Usa a nova busca normalizada ****
            rects = self._search_for_normalized(page, anchor, words_data)
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
                    log.info(f"  ✅(NLP) ÂNCORA (direita): '{anchor}' → '{value[:50]}...'")
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
                    log.info(f"  ✅(NLP) ÂNCORA (abaixo): '{anchor}' → '{value[:50].replace(chr(10), ' ')}...'")
                    return value, heuristic, True
            
            log.info(f"  ✅(NLP) ÂNCORA encontrada ('{anchor}') mas valor está VAZIO.")
            heuristic = Heuristic(
                type="ANCHOR_EMPTY", value=None, confidence=0.8,
                metadata={"anchor_text": anchor, "direction": "right"} 
            )
            return None, heuristic, True

        return None, None, False
    
    def _learn_nlp(self, parsed_pdf: Dict, field: str, description: str) -> Tuple[Optional[str], Optional[Heuristic], bool]:
        # (Este método está correto)
        log.info(f"  🧠(NLP) Tentando aprendizado NLP para: '{field}'")
        
        value, heuristic = self._learn_from_table(parsed_pdf, field, description)
        if heuristic:
            self.stats['nlp_learns'] += 1
            return value, heuristic, True 

        value, heuristic, found = self._learn_from_anchor(parsed_pdf, field, description)
        if found: 
            self.stats['nlp_learns'] += 1
            return value, heuristic, True
        
        log.info(f"  ℹ️(NLP) Não encontrou padrão para '{field}'")
        return None, None, False

    # ==================== NÍVEL 3: APRENDIZADO LLM ====================
    
    def _call_llm(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        # (Este método está correto)
        try:
            log.info(f"  🤖 Chamando LLM... (JSON Mode: {json_mode})")
            
            model_params = {
                "model": LLM_MODEL_NAME, "messages": [
                    {"role": "system", "content": "Você é um assistente especialista em extração de dados de documentos. Siga as instruções de formato da resposta com precisão."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1.0
            }
            if json_mode:
                model_params["response_format"] = {"type": "json_object"}

            response = self.llm_client.chat.completions.create(**model_params)
            log.info(f"  🤖 Terminado LLM... (JSON Mode: {json_mode})")
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * self.INPUT_COST) + (output_tokens * self.OUTPUT_COST)
            self.stats['total_cost'] += cost
            
            log.info(f"  💰 Custo: ${cost:.6f} ({input_tokens} in + {output_tokens} out)")
            
            content = response.choices[0].message.content
            return content.strip() if content else None
            
        except Exception as e:
            log.error(f"  ❌ Erro ao chamar LLM: {e}")
            return None
    
    # ATUALIZADO: _derive_heuristic_for_value (Tenta Âncora, depois Zona)
    def _derive_heuristic_for_value(self, parsed_pdf: Dict, field: str, value: str) -> Optional[Heuristic]:
        page = parsed_pdf["page"]
        words_data = parsed_pdf.get("words")
        if not value: return None
        
        # --- 1. Encontra a Localização do Valor ---
        clean_value = value.strip().replace(',', ' ').replace('\n', ' ')
        value_parts = clean_value.split()
        if len(value_parts) == 0:
            return None
        
        value_for_search = " ".join(value_parts[0:3]) # Tenta 3 palavras
        value_rects = self._search_for_normalized(page, value_for_search, words_data)
        
        if not value_rects:
            value_for_search = " ".join(value_parts[0:1]) # Fallback: 1 palavra
            value_rects = self._search_for_normalized(page, value_for_search, words_data)
            
            if not value_rects:
                log.warning(f"  ⚠️  Não foi possível encontrar o texto (normalizado) '{value_for_search}' no PDF para derivar heurística.")
                return None
        
        rect = value_rects[0]
        
        # --- 2. TENTA DERIVAR UMA ÂNCORA (MUITO PREFERÍVEL) ---
        
        # Tenta Âncora (acima)
        anchor_rect_above = fitz.Rect(rect.x0 - 50, max(0, rect.y0 - 50), rect.x1 + 50, rect.y0 - 2)
        anchor_text_above = page.get_text("text", clip=anchor_rect_above).strip()
        if anchor_text_above:
            anchor = anchor_text_above.split('\n')[-1].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"  🧠 Heurística ÂNCORA (acima) derivada: '{anchor}' → '{value[:20]}...'")
                return Heuristic(
                    type="ANCHOR", value=None, confidence=0.9,
                    metadata={"anchor_text": anchor, "direction": "below"}
                )

        # Tenta Âncora (esquerda)
        anchor_rect_left = fitz.Rect(max(0, rect.x0 - 300), rect.y0 - 5, rect.x0 - 2, rect.y1 + 5)
        anchor_text_left = page.get_text("text", clip=anchor_rect_left).strip()
        if anchor_text_left:
            anchor = anchor_text_left.split('\n')[-1].strip().rstrip(' :')
            if len(anchor) > 3:
                log.info(f"  🧠 Heurística ÂNCORA (esquerda) derivada: '{anchor}' → '{value[:20]}...'")
                return Heuristic(
                    type="ANCHOR", value=None, confidence=0.9,
                    metadata={"anchor_text": anchor, "direction": "right"}
                )
        
        # --- 3. FALLBACK: CRIAR UMA ZONA "HORIZONTAL SLICE" ---
        # (Se nenhuma âncora foi encontrada, como no campo 'nome')
        
        page_rect = page.rect
        
        # Usa a posição Y do 'rect', mas a LARGURA TOTAL da página.
        y0 = max(0, rect.y0 - 5)
        y1 = min(page_rect.height - 2, rect.y1 + 5) 
        x0 = 5.0 # Margem esquerda
        x1 = page_rect.width - 5.0 # Margem direita

        # Lógica para multi-linha (como um endereço)
        if '\n' in value or len(clean_value) > 80:
            y1 = min(page_rect.height - 2, rect.y1 + 70) # Expande para baixo
            log.info("  🧠 Heurística ZONA (multi-linha) detectada.")
        
        zone_coords = [x0, y0, x1, y1]
        log.info(f"  🧠 FALLBACK: Nenhuma âncora encontrada. Heurística ZONA 'Horizontal Slice' salva: {zone_coords}")
        
        return Heuristic(type="ZONE", value=zone_coords, confidence=0.7, metadata={})

    def _llm_fallback_for_field(self, parsed_pdf: Dict, field: str, description: str) -> Tuple[Optional[str], Optional[Heuristic], bool]:
        log.info(f"  🚨(LLM) Fallback: '{field}'")
        text = parsed_pdf["full_text"] # Usa o texto com layout
        prompt = f"""Extraia o valor do seguinte campo do documento:
        Campo: {field}
        Descrição: {description}
        Documento (Texto com Layout):
        ---
        {text}
        ---
        Responda APENAS com o valor encontrado.
        Se não encontrar, responda: null"""
        
        value = self._call_llm(prompt, json_mode=False)
        self.stats['llm_fallbacks'] += 1 # Contagem movida para _batch_llm_fallback
        
        if not value or "null" in value.lower() or "NÃO ENCONTRADO" in value.upper():
            log.warning(f"  ⚠️(LLM) Fallback não encontrou '{field}'")
            return None, None, True 
        
        value = value.strip().strip('"\'')
        log.info(f"  ✅(LLM) Fallback encontrou: '{value}'")
        heuristic = self._derive_heuristic_for_value(parsed_pdf, field, value)
        return value, heuristic, True

    def _bootstrap_new_label_with_llm(self, label: str, extraction_schema: Dict[str, str], parsed_pdf: Dict) -> Dict[str, Any]:
        log.info(f"  🚀(LLM) Bootstrap: Label novo '{label}'. Chamando LLM para dados e classificação.")
        self.stats['llm_bootstraps'] += 1
        
        fields_json_str = json.dumps(extraction_schema, indent=2, ensure_ascii=False)
        text = parsed_pdf["full_text"] # Usa o texto com layout
        
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

        Retorne APENAS o objeto JSON completo. Exemplo:
        {{
          "template_fixo": true,
          "fields": {{
            "nome": "JOAO DA SILVA",
            "inscricao": "123.456",
            "endereco_profissional": "Rua Exemplo, 123\nBairro Centro\nCidade-UF"
          }}
        }}
        """
        
        print(text)
        response_str = self._call_llm(prompt, json_mode=True)
        print(f"response LLM: {response_str}")
        
        if not response_str:
            log.error("  ❌(LLM) Bootstrap falhou. Nenhuma resposta do LLM.")
            return {field: None for field in extraction_schema.keys()}
        
        try:
            response_json = json.loads(response_str)
            
            # 1. Salva a classificação do template
            is_fixed = response_json.get('template_fixo', True)
            self.label_metadata[label] = {'template_fixo': is_fixed}
            log.info(f"  ℹ️  Template classificado como: {'FIXO' if is_fixed else 'VARIÁVEL'}")

            final_data_clean = {} # Para o usuário (com ', ')
            self.KB[label] = {}
            
            # 2. Processa os campos
            fields_data = response_json.get("fields", {})
            for field in extraction_schema.keys():
                
                value_str_raw = str(fields_data.get(field)).strip() if fields_data.get(field) else None
                
                if not value_str_raw or value_str_raw.lower() == 'null':
                    log.info(f"  ℹ️(LLM) Bootstrap: '{field}' → NÃO ENCONTRADO")
                    final_data_clean[field] = None
                    # Tenta aprender heurística de campo vazio (ANCHOR_EMPTY)
                    _, heuristic, found = self._learn_from_anchor(parsed_pdf, field, extraction_schema[field])
                    if found and heuristic and is_fixed:
                        self.KB[label][field] = heuristic
                        log.info(f"  💾 Heurística NLP ('{heuristic.type}') salva para campo Nulo!")
                    continue

                log.info(f"  ✅(LLM) Bootstrap: '{field}' → '{value_str_raw[:50].replace(chr(10), ' ')}...'")
                
                # O CÓDIGO SEMPRE DERIVA A HEURÍSTICA
                heuristic = self._derive_heuristic_for_value(parsed_pdf, field, value_str_raw)

                if heuristic and is_fixed:
                    self.KB[label][field] = heuristic
                    log.info(f"  💾 Heurística {heuristic.type} (derivada) salva para '{field}'")
                elif not heuristic and is_fixed:
                     log.warning(f"  ⚠️  Não foi possível derivar heurística para '{field}' (Valor: '{value_str_raw}')")
                
                # Limpa o valor para o usuário
                value_str_clean = re.sub(r'\s*\n\s*', ', ', value_str_raw)
                final_data_clean[field] = value_str_clean

            return final_data_clean
            
        except json.JSONDecodeError as e:
            log.error(f"  ❌(LLM) Falha ao decodificar JSON do Bootstrap: {e}")
            log.error(f"  Resposta recebida: {response_str}")
            return {field: None for field in extraction_schema.keys()}
        except Exception as e:
            log.error(f"  ❌(LLM) Erro inesperado no Bootstrap: {e}")
            return {field: None for field in extraction_schema.keys()}

    # ATUALIZADO: Método para extração "rápida" de templates variáveis
    def _extract_variable_template(self, extraction_schema: Dict[str, str], parsed_pdf: Dict) -> Dict[str, Any]:
        """
        Extrai dados de um template variável (ex: app screenshot).
        Sempre chama o LLM (gpt-5-mini), mas usa um prompt
        para uma resposta mais rápida e barata.
        """
        log.info(f"  🤖(LLM) Template variável: Chamando LLM com prompt rápido.")
        self.stats['llm_fallbacks'] += 1 # Conta como um fallback
        
        fields_json_str = json.dumps(extraction_schema, indent=2, ensure_ascii=False)
        
        # --- OTIMIZAÇÃO DE VELOCIDADE/CUSTO ---
        # Trunca o texto. Se for um screenshot ou template variável,
        # 8000 caracteres (aprox. 2000 tokens) é mais que suficiente
        # e garante uma resposta muito mais rápida do LLM.
        text = parsed_pdf["full_text"]
        
        # Prompt simplificado para velocidade
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
            
            # Limpa o \n da saída
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

    # NOVO: Método para enviar falhas em lote para o LLM
    def _batch_llm_fallback(self, fallback_queue: List[Dict], requests: List[ExtractRequest]) -> Dict[str, Any]:
        """
        Envia uma lista de falhas de heurística para o LLM numa única chamada.
        """
        
        prompt_parts = [
            "Você é um especialista em extração de dados. Eu tenho vários documentos e preciso extrair campos específicos que falharam na primeira tentativa.",
            "Aqui estão os textos dos documentos (com layout preservado):\n"
        ]
        
        # 1. Adiciona os textos dos documentos (apenas os necessários)
        doc_texts = {}
        for item in fallback_queue:
            req_index = item['req_index']
            if req_index not in doc_texts:
                # Obtém o texto do 'parsed_pdf' guardado no objeto ExtractRequest
                doc_texts[req_index] = requests[req_index]._parsed_pdf['full_text']
        
        for req_index, text in doc_texts.items():
            prompt_parts.append(f"--- DOCUMENTO {req_index} ---")
            prompt_parts.append(text)
            prompt_parts.append(f"--- FIM DOCUMENTO {req_index} ---\n")
        
        # 2. Adiciona os pedidos de extração
        prompt_parts.append("Agora, por favor, extraia os seguintes campos e retorne um único objeto JSON. A chave do JSON deve ser \"doc_INDEX_CAMPO\".")
        
        for item in fallback_queue:
            req_index = item['req_index']
            field = item['field']
            description = item['description']
            # Cria uma chave única global para cada pedido
            key = f"doc_{req_index}_{field}" 
            prompt_parts.append(f"- {key}: (Descrição: {description})")
            
        prompt_parts.append("\nRetorne APENAS o objeto JSON. Use 'null' se não encontrar um valor. Exemplo:")
        prompt_parts.append("""
{
  "doc_0_campo_A": "Valor A do doc 0",
  "doc_1_campo_B": "Valor B do doc 1",
  "doc_1_campo_C": null
}
""")
        
        final_prompt = "\n".join(prompt_parts)
        
        # 3. Faz a chamada LLM única
        response_str = self._call_llm(final_prompt, json_mode=True)
        # Contamos como uma única chamada de fallback para estatísticas
        self.stats['llm_fallbacks'] += 1 
        
        if not response_str:
            log.error("  ❌(LLM) Falha catastrófica no Lote LLM. Nenhuma resposta.")
            return {}
        
        try:
            batch_results = json.loads(response_str)
            return batch_results
        except json.JSONDecodeError as e:
            log.error(f"  ❌(LLM) Falha ao decodificar JSON do Lote LLM: {e}")
            log.error(f"  Resposta recebida: {response_str}")
            return {}


    # ==================== EXTRAÇÃO PRINCIPAL ====================
    
    def print_stats(self):
        # (Este método está correto)
        print("\n" + "="*70)
        print("📊 ESTATÍSTICAS DO EXTRATOR")
        print("="*70)
        print(f"💾 Cache hits:           {self.stats['cache_hits']}")
        print(f"📊 Extração por tabela:  {self.stats['table_extractions']}")
        print(f"🔗 Extração por âncora:  {self.stats['anchor_extractions']}")
        print(f"📍 Extração por zona:    {self.stats['zone_extractions']}")
        print(f"🧠 Aprendizado NLP:      {self.stats['nlp_learns']}")
        print(f"🚀 LLM Bootstraps (1ª vez): {self.stats['llm_bootstraps']}")
        print(f"🚨 LLM Fallbacks (lote): {self.stats['llm_fallbacks']}")
        print(f"💰 Custo total:          ${self.stats['total_cost']:.6f}")
        print(f"📚 Labels aprendidos:    {len(self.KB)}")
        
        total_heuristics = sum(len(fields) for fields in self.KB.values())
        print(f"🎯 Heurísticas salvas:   {total_heuristics}")
        print("="*70 + "\n")
    
    def export_kb(self, filepath: str):
        # (Este método está correto)
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
        
        log.info(f"✅ Knowledge base exportado para: {filepath}")
    
    def import_kb(self, filepath: str):
        # (Este método está correto)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.label_metadata = data.get('metadata', {})
            kb_data = data.get('kb', data)
            
            for label, fields in kb_data.items():
                self.KB[label] = {}
                for field, heur_data in fields.items():
                    self.KB[label][field] = Heuristic(
                        type=heur_data['type'],
                        value=heur_data['value'],
                        confidence=heur_data.get('confidence', 1.0),
                        metadata=heur_data.get('metadata', {})
                    )
            
            log.info(f"✅ Knowledge base importado de: {filepath} (Labels: {len(self.KB)}, Metadados: {len(self.label_metadata)})")
            
        except Exception as e:
            log.error(f"❌ Erro ao importar KB: {e}")

    # NOVO: Método principal de extração em lote
    def extract_batch(self, requests: List[ExtractRequest]) -> List[Dict[str, Any]]:
        """
        Processa uma lista de pedidos de extração em lote para otimizar chamadas LLM.
        """
        
        fallback_queue = [] # Armazena { 'req_index': int, 'field': str, 'description': str }
        
        # --- FASE 1: HEURÍSTICAS E CHAMADAS NÃO-LOTEÁVEIS ---
        log.info(f"🚀 Iniciando extração em lote para {len(requests)} documentos...")
        for i, req in enumerate(requests):
            log.info(f"  [Lote {i+1}/{len(requests)}] Processando (Heurísticas): {req.pdf_path}")
            
            # 1. Verificar Cache de PDF
            req._pdf_hash = self._get_pdf_hash(req.pdf_path)
            if req._pdf_hash and req._pdf_hash in self.pdf_cache:
                log.info(f"    ✅ CACHE HIT (PDF já processado)")
                self.stats['cache_hits'] += 1
                req._result = self.pdf_cache[req._pdf_hash]
                continue # Este pedido está concluído

            # 2. Parsear PDF
            req._parsed_pdf = self._parse_pdf(req.pdf_path)
            if not req._parsed_pdf:
                log.error(f"    ❌ Falha ao parsear PDF, pulando.")
                req._result = {field: None for field in req.extraction_schema.keys()}
                continue # Este pedido falhou

            # 3. Verificar Bootstrap (Não pode ser loteado, custo único)
            if req.label not in self.KB:
                log.info(f"    📝 Label novo: '{req.label}'. Iniciando Bootstrap (chamada individual)...")
                # Bootstrap retorna dados finais limpos (com ', ')
                bootstrap_data_clean = self._bootstrap_new_label_with_llm(
                    req.label, req.extraction_schema, req._parsed_pdf
                )
                # Converte de volta para dados brutos (com \n) para o resultado e cache
                req._result = {f: (v.replace(', ', '\n') if isinstance(v, str) else None) for f, v in bootstrap_data_clean.items()}
                if req._pdf_hash:
                    self.pdf_cache[req._pdf_hash] = req._result.copy()
                continue # Este pedido está concluído

            # 4. Verificar Template Variável (Não pode ser loteado)
            is_fixed = self.label_metadata.get(req.label, {}).get('template_fixo', True)
            if not is_fixed:
                log.info("    >> Estratégia: Template Variável (Chamada LLM individual)...")
                # Retorna dados limpos (com ', ')
                variable_data_clean = self._extract_variable_template(
                    req.extraction_schema, req._parsed_pdf
                )
                # Converte de volta para dados brutos (com \n) para o resultado e cache
                req._result = {f: (v.replace(', ', '\n') if isinstance(v, str) else None) for f, v in variable_data_clean.items()}
                if req._pdf_hash:
                    self.pdf_cache[req._pdf_hash] = req._result.copy()
                continue # Este pedido está concluído
            
            # 5. Executar Heurísticas de Template Fixo
            log.info("    >> Estratégia: Template Fixo (Apenas Heurísticas)...")
            for field, description in req.extraction_schema.items():
                value = None
                found = False
                
                if field in self.KB[req.label]:
                    heuristic = self.KB[req.label][field]
                    value, found = self._apply_heuristic(heuristic, req._parsed_pdf)
                
                # Lógica de fallback: falha se não encontrado OU se for 'ANCHOR_EMPTY'
                if found and value is not None:
                    # Sucesso real
                    req._result[field] = value # Guarda o valor bruto (com \n)
                else:
                    # HEURÍSTICA FALHOU OU RETORNOU NULO (ex: ANCHOR_EMPTY)!
                    if found and value is None:
                        log.warning(f"    ⚠️  Heurística 'ANCHOR_EMPTY' retornou Nulo para '{field}'. Verificando com LLM.")
                    else:
                        log.warning(f"    ⚠️  Heurística falhou para '{field}'. Adicionando à fila de LLM.")
                    
                    req._failed_fields.append(field)
                    fallback_queue.append({
                        'req_index': i,
                        'field': field,
                        'description': description
                    })

        # --- FASE 2: FALLBACK LLM EM LOTE ---
        if not fallback_queue:
            log.info("✅ Lote concluído. Nenhuma chamada de LLM fallback necessária.")
        else:
            log.info(f"🚨 Processando {len(fallback_queue)} falhas de heurística em um único lote LLM...")
            
            # Chama o novo método de lote
            batch_results = self._batch_llm_fallback(fallback_queue, requests)
            
            # Distribuir os resultados de volta para os pedidos
            for item in fallback_queue:
                req_index = item['req_index']
                field = item['field']
                
                # A chave única que criámos
                key = f"doc_{req_index}_{field}"
                value = batch_results.get(key) # Valor bruto (pode ter \n)
                
                if value and value.lower() != 'null':
                    log.info(f"    [Lote {req_index+1}] ✅ LLM Fallback SUCESSO para '{field}'")
                    requests[req_index]._result[field] = value
                    
                    # ✨ AUTOCORREÇÃO! ✨
                    new_heuristic = self._derive_heuristic_for_value(
                        requests[req_index]._parsed_pdf, field, value
                    )
                    if new_heuristic:
                        self.KB[requests[req_index].label][field] = new_heuristic
                        log.info(f"    [Lote {req_index+1}] 💾 AUTOCORREÇÃO: Nova heurística '{new_heuristic.type}' salva para '{field}'")
                else:
                    log.warning(f"    [Lote {req_index+1}] ❌ LLM Fallback FALHA para '{field}'")
                    requests[req_index]._result[field] = None

        # --- FASE 3: FINALIZAR E LIMPAR ---
        final_results_list = []
        for req in requests:
            # Garante que todos os campos estão presentes, preenchendo Nones
            final_data_raw = {}
            for field in req.extraction_schema.keys():
                final_data_raw[field] = req._result.get(field)

            # Salvar no cache (dados brutos) se ainda não estiver lá
            if req._pdf_hash and req._pdf_hash not in self.pdf_cache:
                self.pdf_cache[req._pdf_hash] = final_data_raw.copy()

            # Limpar a saída para o utilizador (substituir \n por ', ')
            cleaned_data = {}
            for field, value in final_data_raw.items():
                if isinstance(value, str):
                    cleaned_data[field] = re.sub(r'\s*\n\s*', ', ', value)
                else:
                    cleaned_data[field] = value
            
            final_results_list.append(cleaned_data)
            
            # Fechar o 'doc' do fitz
            if req._parsed_pdf and req._parsed_pdf.get("doc"):
                req._parsed_pdf["doc"].close()

        log.info("✅ Processamento em lote concluído.")
        return final_results_list

    # ATUALIZADO: Método `extract` original agora é um "atalho"
    def extract(self, label: str, extraction_schema: Dict[str, str], 
                pdf_path: str) -> Dict[str, Any]:
        """
        Ponto de entrada para um único ficheiro.
        Envolve o novo método de extração em lote para reutilizar a lógica.
        """
        print(f"\n{'='*70}")
        print(f"🔄 Requisição: {label} | {len(extraction_schema)} campos")
        print(f"📄 PDF: {pdf_path}")
        print(f"{'='*70}")
        
        # 1. Cria um único pedido
        single_request = ExtractRequest(
            label=label,
            extraction_schema=extraction_schema,
            pdf_path=pdf_path
        )
        
        # 2. Chama o processador de lote com uma lista de 1 item
        batch_results = self.extract_batch([single_request])
        
        # 3. Retorna o primeiro (e único) resultado
        return batch_results[0]