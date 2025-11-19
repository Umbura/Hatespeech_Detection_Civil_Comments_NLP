# Neos: NLP Hibrida Paralela para Detecção de Discurso de Ódio(em desenvolvimento)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/NOME_DO_REPOSITORIO/blob/main/Toxicidade_Civil_Comments_NLP.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-F1_Score_0.90-green)

O projeto **Neos** é um modelo de Deep Learning de alta performance desenvolvido por mim, para a detecção e classificação de discurso de ódio, com foco inicial na distinção entre as classes *"insulto"* e *"ameaça"* do dataset **Civil Comments**.

A principal inovação do Neos reside em sua **arquitetura híbrida paralela CNN + Bi-LSTM ou GRU e na decisão estratégica de **não utilizar embeddings pré-treinados**, permitindo que o modelo aprenda gírias e termos ofuscados específicos do vocabulário tóxico.

## Evolução do Projeto "Neos"

O desenvolvimento seguiu uma metodologia iterativa baseada em falhas e aprendizados:

### 1. Fase Inicial (Protótipo OvR)
*   **Abordagem:** Classificação em dois estágios *One-vs-Rest (OvR)*, inspirada em *Pitsilis et al. (2022)*.
*   **Arquitetura:** Múltiplos classificadores Bi-LSTM independentes.
*   **Resultado:** Acurácia de ~71% (Abaixo do esperado).
*   **Problema:** Complexidade de gerenciamento de múltiplos modelos e baixa performance na integração das probabilidades.

### 2. Fase Intermediária (Híbrido Sequencial)
*   **Abordagem:** Extração de features com CNN seguida de processamento sequencial (CNN -> RNN).
*   **Resultado:** Acurácia de ~86%.
*   **Limitação Crucial:** A camada CNN filtrava excessivamente o texto, fazendo com que a camada recorrente subsequente perdesse informações contextuais vitais da frase original.

### 3. Fase Intermediária (Neos: Híbrido Paralelo)
*   **Solução:** Processamento simultâneo do texto em dois ramos distintos que convergem no final.
*   **Resultado:** Superou a meta de 90% de acurácia, atingindo **F1-Score de ~0.91** em duas classes threat e insult.
![Neos.v3.0]()

## Arquitetura Final (em andamento)

O modelo atua como um "comitê de especialistas", processando a entrada de texto simultaneamente em dois ramos:

1.  **Input & Embedding:** Camada de Embedding treinada do zero (sem pesos pré-treinados como GloVe ou Word2Vec) para capturar nuances específicas do dataset.
2.  **Ramo Contextual (Bi-LSTM ou GRU):** Foca no entendimento do contexto e na ordem sequencial das palavras.
3.  **Ramo de Padrões (Multi-Kernel CNN):** Foca na detecção de *n-gramas* e palavras-chave tóxicas (padrões locais), independentemente de sua posição.
4.  **Fusão e Classificação:** As saídas dos dois ramos são concatenadas, oferecendo uma visão rica e diversificada para a camada densa final.

## Engenharia de Dados e Treinamento

Para garantir a robustez dos resultados relatados:

*   **Dataset:** [Jigsaw/Civil Comments](https://huggingface.co/datasets/civil_comments).
*   **Balanceamento:** Utilização de **Subamostragem Aleatória (Undersampling)** nas classes majoritárias e **Oversampling** nas minoritárias para garantir um dataset perfeitamente equilibrado.
*   **Validação:** Validação Cruzada Estratificada (*Stratified K-Fold*) de 5 folds.
*   **Otimização:** Uso de *Early Stopping* para prevenir overfitting.

## Resultados Atuais

| Métrica | Resultado Médio | Meta |
| :--- | :--- | :--- |
| **F1-Score** | **0.90** | 0.93 |

> *Nota do Autor: O resultado de 0.90 é robusto, mas o projeto continua ativo com o objetivo de refinar a arquitetura para alcançar 0.93, além de validar o modelo em datasets em Português-BR.*

## Referências Bibliográficas

Este projeto foi fundamentado nas seguintes obras:

1.  **PITSILIS, G. K.** *Improved two-stage hate speech classification for twitter based on Deep Neural Networks*. arXiv preprint arXiv:2206.04162, 2022.
2.  **ZHOU, C. et al.** *A C-LSTM Neural Network for Text Classification*. COLING 2016.
3.  **SCHUSTER, M.; PALIWAL, K. K.** *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 1997.
4.  **JIGSAW/GOOGLE.** *Jigsaw Unintended Bias in Toxicity Classification*. Kaggle, 2019.

## Como Executar

### Requisitos
*   Python 3.8+
*   TensorFlow 2.x
*   Bibliotecas listadas em `requirements.txt`

### Instalação
```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
pip install -r requirements.txt
