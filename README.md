# Projeto Lucro Bruto: M5 Forecasting Accuracy

## 1. Contexto e Problema de Negócio
No varejo de alta escala, a ineficiência no planejamento de estoque impacta diretamente o fluxo de caixa e o capital de giro. Este projeto ataca duas frentes principais de perda de margem:

- **Ruptura de Estoque**  
  Perda de receita e custo de oportunidade devido à indisponibilidade de produtos.

- **Excesso de Estoque**  
  Capital imobilizado, custos logísticos de armazenagem e aumento do risco de perdas físicas.

**Desafio:**  
Transicionar de uma análise reativa para uma operação preditiva, considerando que variações de apenas **5% no erro de previsão** são críticas para a saúde financeira.

---

## 2. Objetivos Estratégicos
O motor de inteligência utiliza o dataset **Walmart M5** para entregar previsões granulares com foco em:

- **Identificação da Cauda Longa (Pareto)**  
  Priorizar os 20% dos produtos responsáveis por 80% do lucro bruto.

- **Inteligência Hierárquica**  
  Modelar relações entre:
  - Unidades (SKUs)
  - Lojas
  - Departamentos
  - Categorias

- **Tradução Financeira**  
  Converter métricas estatísticas (ex: erro de previsão) em impacto direto no **Lucro Bruto (R$)**.

---

## 3. Mapa de Batalha (Metodologia)

### 3.1 Entendimento da Hierarquia
- Análise de sazonalidade  
- Identificação de tendências  
- Separação de ruído vs. sinal  

### 3.2 Engenharia de Features
- Criação de **lags (defasagens)**  
- Janelas móveis (rolling windows)  
- Variáveis de calendário:
  - Feriados
  - Eventos promocionais  

### 3.3 Validação Robusta
- Uso da métrica **WRMSSE (Weighted Root Mean Squared Scaled Error)**  
- Priorização de erros em itens de maior impacto econômico  

---

## 4. Stack Técnica & Engenharia

#### Pipeline de Dados
Arquitetura de medalhões:

- **Bronze** → Dados brutos (imutáveis)  
- **Silver** → Dados tratados e enriquecidos  
- **Gold** → Dados prontos para consumo analítico  

#### Orquestração Local
- Uso de `pathlib` e `shutil`  
- Automação do ciclo de vida dos arquivos  
- Garantia de imutabilidade na camada Bronze  

### EDA Orientada a Valor
- Análise de séries temporais de alta granularidade  
- Identificação de dispersão  
- Separação entre ruído e padrão relevante  

---

## 5. Estrutura do Repositório

```bash
/app        # Interfaces e módulos de produção (Streamlit / FastAPI)
/data       # Camadas Bronze, Silver e Gold
/notebooks  # Exploração, hipóteses e validação
```

## 6. Resultado Esperado

- Redução do erro de previsão  
- Otimização do capital de giro  
- Aumento direto do **Lucro Bruto**  
- Base para evolução de modelos preditivos escaláveis  