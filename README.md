# 🚀 Smart PDF Extractor

Esta é uma solução de extração de dados de PDFs que implementa uma arquitetura de auto-aprendizagem para otimizar o equilíbrio entre custo, velocidade e precisão.

O sistema aprende com a primeira extração de um novo tipo de documento (definido por um label) e cria heurísticas (Âncoras e Zonas) para tornar todas as extrações futuras desse mesmo label instantâneas e **sem custo de LLM**.

## 🧠 Desafios Mapeados e Soluções Propostas

O desafio principal não é apenas extrair dados, mas fazê-lo de forma eficiente, lidando com layouts fixos e variáveis, aprendendo com o tempo e otimizando o custo de múltiplas chamadas de API.

**Desafio 1: Custo vs. Precisão (O Dilema do LLM)**

* **Desafio**: Chamar um LLM (como o `gpt-5-mini`) para cada PDF é caro, lento e um desperdício, especialmente para documentos com layouts fixos (como a `carteira_oab`).

* **Solução Proposta**: Uma arquitetura de auto-aprendizagem baseada em label.

   * O LLM (o "cérebro" caro) é usado **apenas uma vez** por label, na função _bootstrap_new_label_with_llm.

   * Esta função usa o LLM para extrair os dados e também para classificar o template como `"template_fixo": true` ou `false`.

      Se for fixo, o sistema aprende e salva heurísticas (Âncoras e Zonas) num ficheiro `knowledge_base.json`.

   * Todas as extrações futuras com o mesmo label **usam estas heurísticas de custo zero, tornando-as quase instantâneas**.

**Desafio 2: Layouts "Quebradiços" e Otimização de Velocidade**

* **Desafio**: Heurísticas são frágeis. Uma âncora pode falhar por causa de um acento (`Inscrição` vs `Inscricao`), e uma zona pode cortar palavras (`LUIS FILIPE A` em vez de `LUIS FILIPE ARAUJO AMARAL`).

* **Solução Proposta**:

   1. **Busca Normalizada Rápida**: Foi criada uma função `_search_for_normalized` que pré-processa todas as palavras do PDF uma vez (`parsed_pdf_cache`). Esta busca ignora acentos e capitalização (ex: 'ç' == 'C'), tornando as âncoras 99% mais robustas.

   2. **Autocorreção Híbrida (Âncora/Zona)**: A função de aprendizagem `_derive_heuristic_for_value` é inteligente. Ela primeiro tenta encontrar uma âncora robusta (ex: "Inscrição" acima de "101943"). Se falhar (ex: para um campo "nome" no topo da página), ela cria como fallback uma `ZONE` "Horizontal Slice" (fatia horizontal), que usa a largura total da página para garantir que não corta palavras.

   3. **Gestão de Nulos**: O sistema aprende a regra `ANCHOR_EMPTY` para campos que existem mas estão vazios (ex: `telefone_profissional`). Isto evita chamar o LLM desnecessariamente para campos nulos.

**Desafio 3: Processamento em Lote Eficiente**

* **Desafio**: Processar 1000 PDFs onde 10% falham numa heurística significaria 100 chamadas de LLM separadas, o que é lento e caro.

* **Solução Proposta**: O método extract_batch.

   1. O sistema primeiro tenta extrair tudo usando as heurísticas de Custo Zero (Nível 3).

   2. Todos os campos que falham (em todos os PDFs) são adicionados a uma única fallback_queue.

   3. No final, o _batch_llm_fallback é chamado uma única vez, enviando todas as falhas num "prompt massivo" para o LLM.

   4. Os resultados são usados para a autocorreção (Nível 2), melhorando o KB para o futuro.

## 🚀 Como Utilizar

A solução é entregue como uma aplicação web Flask (app.py) que serve uma UI simples (index.html) e expõe endpoints de API. O processamento em lote é feito através de um script de cliente (batch_extract.py).

1. Pré-requisitos
   * Python 3.10+
   * Chave da API da OpenAI

2. Instalação

   * Clone o repositório.

   * Crie um ambiente virtual: python -m venv venv e source venv/bin/activate

   * Instale as dependências: pip install -r requirements.txt

   * Defina sua chave de API (escolha uma):

      * Método A (Bash): `export OPENAI_API_KEY='sk-...'`

      * Método B (.env): Crie um arquivo `.env` no diretório e adicione `OPENAI_API_KEY='sk-...'`.

3. Executando a Aplicação (UI + API)

   ```Bash
   python app.py
   ```
   A aplicação estará disponível em http://127.0.0.1:8000.

**4. Usando a Interface Web (UI)**

Aceda a http://127.0.0.1:8000 no seu navegador. A UI tem três abas:

   * **Extração Única**: Permite enviar um único PDF com um label e schema para teste rápido.

   * **Extração em Lote**: Permite fazer o upload de um ficheiro JSON de pedidos e dos múltiplos PDFs correspondentes, e depois baixar os resultados.

   * **Instruções da API**: Mostra exemplos curl para usar a API diretamente.

**5. Usando o Script de Lote (Recomendado)**

Esta é a forma mais poderosa de usar a solução.

1. **Crie o seu ficheiro de pedidos** (ex: `example_requests.json`): O `pdf_path` é o **ID** (nome do ficheiro) que o script irá procurar na sua pasta de PDFs.

```JSON

[
  {
    "label": "carteira_oab",
    "extraction_schema": { "nome": "Nome do profissional", "inscricao": "Número" },
    "pdf_path": "oab_1.pdf"
  },
  {
    "label": "tela_sistema",
    "extraction_schema": { "produto": "Produto da operação" },
    "pdf_path": "tela_1.pdf"
  },
  {
    "label": "carteira_oab",
    "extraction_schema": { "nome": "Nome", "seccional": "Seccional" },
    "pdf_path": "oab_2.pdf"
  }
]
```

2. **Coloque os seus PDFs numa pasta**:

```bash
/meus_pdfs/
├── oab_1.pdf
├── oab_2.pdf
└── tela_1.pdf
```

3. **Execute o script**: O script `batch_extract.py` envia o JSON e todos os PDFs encontrados na pasta para a API `/api/batch_upload`.

```Bash
# python batch_extract.py [CAMINHO_JSON] [CAMINHO_PASTA_PDFS]
python batch_extract.py ./example_requests.json ./meus_pdfs/
```

**4. Receba os Resultados**: 

O script irá imprimir os resultados no terminal e também salvará automaticamente um ficheiro `extraction_results.json` no seu diretório.