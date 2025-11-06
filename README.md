# 🚀 Smart PDF Extractor

Esta é uma solução de extração de dados de PDFs que implementa uma arquitetura de auto-aprendizagem para otimizar o equilíbrio entre custo, velocidade e precisão.

O sistema aprende com a primeira extração de um novo tipo de documento (definido por um label) e cria heurísticas (Âncoras, Zonas e Regex) para tornar todas as extrações futuras desse mesmo `label` instantâneas e **sem custo de LLM**.

É importante notar que a primeira requisição de um novo `label` (Nível 1) sempre será mais lenta, pois exige uma chamada à API do LLM para o aprendizado. No entanto, as requisições subsequentes para esse label tendem a ser processadas localmente (Nível 2), tornando-as quase instantâneas ou com menos informações a serem processadas no llm (maior rapidez), o que permite que a média de processamento em lote atenda ao objetivo de <10s por documento.

## 🧠 Arquitetura de Auto-Aprendizagem

O desafio principal não é apenas extrair dados, mas fazê-lo de forma eficiente, lidando com layouts fixos e variáveis, aprendendo com o tempo e otimizando o custo de múltiplas chamadas de API. A arquitetura aprende com o LLM e se autocorrige, operando em três níveis:

### Nível 1: Bootstrap (O Dilema do LLM)

* **Desafio**: Chamar um LLM (como o `gpt-5-mini`) para cada PDF é caro, lento e um desperdício, especialmente para documentos com layouts fixos.

* **Solução Proposta**: Uma arquitetura de auto-aprendizagem baseada em label.

   * O LLM (o "cérebro" caro) é usado **apenas uma vez** por label, na função `_bootstrap_new_label_with_llm`.

   * Esta função usa o LLM para extrair os dados e também para classificar o template como `"template_fixo": true` ou `false`.

      Se for fixo, o sistema aprende e salva heurísticas (Âncora, Zona ou Regex) num ficheiro `knowledge_base.json`.

   * Todas as extrações futuras com o mesmo label **usam estas heurísticas de custo zero, tornando-as quase instantâneas**.

### Nível 2: Extração por Heurística (Custo Zero)

* **Desafio**: Heurísticas são frágeis. Uma âncora pode falhar por causa de um acento (`Inscrição` vs `Inscricao`), e uma zona pode cortar palavras (`LUIS FILIPE A` em vez de `LUIS FILIPE ARAUJO AMARAL`).

* **Solução Proposta**: O sistema usa um conjunto de heurísticas inteligentes e robustas:

   1. **Busca Normalizada Rápida**: Foi criada uma função `_search_for_normalized` que pré-processa todas as palavras do PDF uma vez (`parsed_pdf_cache`). Esta busca ignora acentos e capitalização (ex: 'ç' == 'C'), tornando as âncoras 99% mais robustas.

   2. **Aprendizado de Âncora (4 Direções)**: A função `_derive_heuristic_for_value` não procura âncoras só "acima" ou "à esquerda", mas em todas as 4 direções (acima, abaixo, esquerda, direita), tornando a heurística de `ANCHOR` muito mais provável de ser encontrada.

   3. **Aprendizado de Regex**: Ao aprender, `_guess_regex_for_value` tenta adivinhar um padrão para o valor (ex: `\d{2}/\d{2}/\d{4}` para datas, `\d{3}\.\d{3}\.\d{3}-\d{2}` para CPFs).

   4. **Extração Precisa (com Regex)**: Se um Regex foi aprendido, as funções `_apply_anchor_heuristic` e `_apply_zone_heuristic` o utilizam para filtrar o texto extraído, garantindo que apenas o valor no formato correto seja retornado.

   5. **Gestão de Nulos**: O sistema aprende a regra `ANCHOR_EMPTY` para campos que existem mas estão vazios (ex: telefone_profissional), evitando chamar o LLM desnecessariamente.

### Nível 3: Fallback e Autocorreção (Processamento Eficiente)

Este nível é acionado quando uma heurística conhecida (Nível 2) falha.

1. O sistema itera por cada PDF individualmente. Para cada um, ele primeiro tenta extrair todos os campos usando as heurísticas de Custo Zero.

2. Todos os campos que falham (em um único PDF) são adicionados a uma lista de falhas temporária (`_failed_fields`) para aquele documento.

3. Se houver qualquer falha, a função `_single_doc_llm_fallback` é chamada uma única vez para aquele PDF.

4. Esta função envia todas as falhas daquele documento (ex: "nome", "inscricao") num "prompt massivo" único para o LLM.

5. Os resultados retornados pelo LLM são usados para a autocorreção, chamando `_derive_heuristic_for_value` para aprender uma nova heurística (agora mais inteligente) e substituindo a antiga no KB.

## 🚀 Como Utilizar

A solução é entregue como uma aplicação web Flask (`app.py`) que serve uma UI simples (index.html) e expõe endpoints de API. O processamento em lote é feito através de um script de cliente (`batch_extract.py`).

1. **Pré-requisitos**
   * Python 3.12+
   * Chave da API da OpenAI

2. **Instalação**

   * Clone o repositório.

   * Crie um ambiente virtual: `python -m venv venv` e `source venv/bin/activate`

   * Instale as dependências: `pip install -r requirements.txt`

   * Defina sua chave de API (escolha uma):

      * Método A (Bash): `export OPENAI_API_KEY='sk-...'`

      * Método B (.env): Crie um arquivo `.env` no diretório e adicione `OPENAI_API_KEY='sk-...'`.

3. **Executando a Aplicação (UI + API)**

   ```Bash
   python app.py
   ```
   A aplicação estará disponível em http://127.0.0.1:8000.

4. **Usando a Interface Web (UI)**

Acesse `http://127.0.0.1:8000` no seu navegador. A UI tem três abas:

   * **Extração Única**: Permite enviar um único PDF com um `label` e `schema` para teste rápido.

   * **Extração em Lote**: Permite fazer o upload de um ficheiro JSON de pedidos e dos múltiplos PDFs correspondentes, e depois baixar os resultados.

   * **Instruções da API**: Mostra exemplos `curl` para usar a API diretamente.

5. **Usando o Script de Lote (Recomendado)**

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

#### ⚠️ Limitações Conhecidas
   Velocidade do LLM: O desempenho do Nível 1 (Bootstrap) e Nível 3 (Fallback) está diretamente atrelado à velocidade de resposta do gpt-5-mini. O modelo é configurado com temperature: 1.0, pois o gpt-5-mini pode não suportar temperature: 0 (o que seria ideal para extração determinística e mais rápida). Isso adiciona latência e variabilidade às operações de aprendizado e autocorreção.

## 🏛️ Arquitetura Alternativa (Com Async e WebSockets)

A arquitetura atual usa `threading` no `app.py` para criar um job em segundo plano e um sistema de polling (consultas repetidas) no endpoint `/api/batch_status/<job_id>` para verificar o progresso. Esta é uma solução robusta e clássica para o Flask.

Se a stack tecnológica permitisse `async` nativo (usando frameworks como FastAPI ou Quart), uma arquitetura ainda mais performática seria possível:

* **API Não-Bloqueante**: O endpoint `/api/batch_upload` seria async e, em vez de threading, usaria BackgroundTasks (FastAPI) ou um sistema de fila dedicado para iniciar o processamento sem bloquear o servidor.

* **Progresso em Tempo Real com WebSockets**: Em vez de o cliente perguntar ao servidor "já terminou?" a cada 2 segundos (polling), o cliente abriria uma conexão WebSocket. O servidor, então, empurraria atualizações de status para o cliente em tempo real (ex: "processado: 5/100", "processado: 6/100"), eliminando a necessidade de polling.

* **Extrator Concorrente**: O SmartExtractor poderia usar um cliente `AsyncOpenAI`. A maior vantagem estaria no `_single_doc_llm_fallback`: se 10 PDFs em um lote precisarem de fallback, em vez de processá-los sequencialmente (esperando 5-10s por cada um), um extrator async poderia executar todas as 10 chamadas de LLM concorrentemente com `asyncio.gather()`, reduzindo o tempo de espera de 100 segundos para ~10 segundos.
