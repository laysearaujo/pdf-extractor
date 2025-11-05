# 🚀 Smart PDF Extractor

**Solução para o Take Home Project da ENTER AI Fellowship**

Esta é uma solução de extração de dados de PDFs que implementa uma arquitetura de roteamento inteligente de 4 níveis para otimizar o trade-off entre **Custo**, **Velocidade** e **Precisão**.

---

## 💡 Desafios e Soluções Propostas

O desafio principal não é apenas extrair dados (uma commodity), mas fazê-lo de forma eficiente, lidando com layouts variáveis e aprendendo com o tempo, tudo isso com restrições de custo (`gpt-5-mini`) e velocidade (<10s).

### Desafio 1: Custo vs. Velocidade (O "Cold Start" de 13s)
- **Problema:** Soluções de RAG Semântico (ex: `SentenceTransformers`) são excelentes para precisão, mas têm um "cold start" de 5-15 segundos para carregar o modelo, quebrando o requisito de <10s para a primeira requisição.
- **Solução:** A arquitetura abandona o RAG Semântico (lento) em favor de um **RAG de Keyword** (rápido, Custo Zero) e **Heurísticas Posicionais** aprendidas dinamicamente.

### Desafio 2: Layouts Variáveis (O problema `carteira_oab` vs `tela_sistema`)
- **Problema:** O mesmo `label` pode ser usado para layouts completamente diferentes (ex: `tela_sistema`), enquanto layouts idênticos podem ter `schemas` ligeiramente diferentes (ex: `carteira_oab`).
- **Solução:** A lógica de roteamento **ignora o `label` como fonte da verdade**. Em vez disso, ela usa um **Roteamento por Similaridade de Schema** (medido pela Similaridade de Jaccard) para decidir qual "template" de heurística aplicar.

### Desafio 3: Minimizar Custo do LLM
- **Problema:** O `gpt-5-mini` tem um custo de output de $2.00/1M tokens, tornando o envio de textos completos inviável.
- **Solução:** Uma arquitetura de roteamento de 4 níveis que SEMPRE tenta uma rota de Custo Zero antes de gastar com o LLM.

---

## 🧠 Arquitetura de Roteamento de 4 Níveis

A classe `SmartExtractor` funciona como um roteador que decide a forma mais barata e rápida de extrair os dados.

### Nível 0: Cache de Hash (Custo Zero, <0.01s)
- **O que faz:** Calcula um hash SHA256 do arquivo PDF. Se o hash já existir em um cache em memória, retorna o resultado salvo instantaneamente.
- **Resolve:** Requisições repetidas do *mesmo* arquivo.

### Nível 1: Cache de Template Posicional (Custo Zero, <0.1s)
- **O que faz:** Quando um `(label, schema)` chega, ele calcula a similaridade com os *templates* já aprendidos.
- **Se Similaridade > 80%:** Aplica uma heurística posicional (um "mapa" de coordenadas X/Y) aprendida com uma chamada de LLM anterior (Nível 3).
- **Resolve:** Layouts fixos (`carteira_oab`) que são vistos repetidamente, mesmo com `schemas` ligeiramente diferentes.

### Nível 1.5: Heurística de Proximidade (Custo Zero, <0.1s)
- **O que faz:** Se nenhum template é encontrado, ele tenta uma heurística "burra" universal: procurar por `Label: Valor` ou `Label\nValor`.
- **Resolve:** Formulários simples (não-posicionais) que nunca foram vistos antes.

### Nível 2: Extração Híbrida (Custo Ultra-Baixo, ~2-3s)
- **O que faz:** Ocorre se o Nível 1 (Posicional) encontrou 80% dos campos, mas 20% estão faltando (ex: um `schema` novo adicionou um campo).
- **Solução:** Roda o **Keyword RAG** *apenas* para os campos faltantes e faz uma chamada "cirúrgica" ao LLM para extrair *apenas* esses campos.
- **Resolve:** Otimização de custo para variações de `schema` em layouts conhecidos (`oab_1` vs `oab_3`).

### Nível 3: LLM Completo (Custo Baixo, ~3-5s)
- **O que faz:** O último recurso. Se Nível 0, 1 e 1.5 falharem (ex: um contrato, ou `tela_sistema` pela primeira vez).
- **Solução:** Roda o **Keyword RAG** no texto completo (para reduzir custo), usa **Few-Shot Learning** (do cache `few_shot_cache`) e chama o `gpt-5-mini` para o `schema` completo.
- **Aprendizado:** O resultado desta chamada é usado para **aprender e salvar um novo Template Posicional (Nível 1)**, tornando a *próxima* extração desse layout instantânea.

---

## 🚀 Como Utilizar

### 1. Pré-requisitos
- Python 3.10+
- Chave da API da OpenAI

### 2. Instalação
1. Clone o repositório.
2. Crie um ambiente virtual: `python -m venv venv` e `source venv/bin/activate`
3. Instale as dependências: `pip install -r requirements.txt`
4. Defina sua chave de API (escolha uma):
   - **Método A (Bash):** `export OPENAI_API_KEY='sk-...'`
   - **Método B (.env):** Crie um arquivo `.env` e adicione `OPENAI_API_KEY='sk-...'`

### 3. Executando a Aplicação (UI + API)
```bash
python app.py
```
