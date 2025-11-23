<div align="center">

# Detecção Híbrida de Discurso de Ódio

### Arquitetura Paralela CNN + Bi-LSTM

<!-- LANGUAGE SWITCHER -->
[![Read in English](https://img.shields.io/badge/Read%20in-English-0077B5?style=for-the-badge&logo=google-translate&logoColor=white)](README.md)

<!-- TECH STACK BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Status-F1_Score_0.90-green" alt="Status F1 0.90">
  <a href="https://colab.research.google.com/github/Umbura/Hatespeech_Detection_Civil_Comments_NLP/blob/main/Toxicidade_Civil_Comments_NLP.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

<!-- MAIN IMAGE -->
<img src="assets/resultado_neos_v4.png" alt="Resultados Neos V4" width="100%">

*Resultados atuais da classificação multiclasse no dataset Civil Comments.*

</div>

---

## Sobre o Projeto
O projeto é um modelo de Deep Learning de alta performance desenvolvido para não apenas detectar, mas classificar o tipo de discurso de ódio praticado. Inicialmente focado nas classes *"insult"* e *"threat"*, o escopo foi expandido para todas as sete classes do dataset **Civil Comments**.

A principal inovação do projeto reside em sua **arquitetura híbrida paralela (CNN + Bi-LSTM)** e na decisão estratégica de **não utilizar embeddings pré-treinados**. Isso permite que o modelo aprenda gírias, neologismos e termos ofuscados específicos do vocabulário tóxico em qualquer idioma, sem depender de dicionários estáticos.

---

## Evolução do Projeto

O desenvolvimento seguiu uma metodologia iterativa baseada em falhas e aprendizados:

### 1. Fase Inicial (Protótipo OvR)
*   **Abordagem:** Classificação em dois estágios *One-vs-Rest (OvR)*, inspirada em *Pitsilis et al. (2022)*.
*   **Arquitetura:** Múltiplos classificadores Bi-LSTM independentes.
*   **Resultado:** Acurácia de ~71% (Abaixo do esperado).
*   **Diagnóstico:** Alta complexidade de gerenciamento de múltiplos modelos e baixa performance na integração das probabilidades.

### 2. Fase Intermediária (Híbrido Sequencial)
*   **Abordagem:** Extração de *features* com CNN seguida de processamento sequencial (CNN → RNN).
*   **Resultado:** Acurácia de ~86%.
*   **Limitação Crucial:** A camada CNN filtrava excessivamente o texto, fazendo com que a camada recorrente subsequente perdesse informações contextuais vitais da frase original.

### 3. Fase Atual (Híbrido Paralelo)
*   **Solução:** Processamento simultâneo do texto em dois ramos distintos que convergem no final.
*   **Resultado:** Superou a meta de 90% de acurácia, atingindo **F1-Score de ~0.91** nas classes críticas (threat e insult).

<div align="center">
  <img src="assets/resultado_neos_v3.png" alt="Evolução V3" width="80%">
</div>

---

## Arquitetura (Em Andamento)

O modelo atua como um "comitê de especialistas", processando a entrada de texto simultaneamente em dois ramos:

1.  **Input & Embedding:** Camada de Embedding treinada do zero (sem pesos como GloVe ou Word2Vec) para capturar nuances específicas do dataset.
2.  **Ramo Contextual (Bi-LSTM):** Foca no entendimento do contexto e na ordem sequencial das palavras.
3.  **Ramo de Padrões (Multi-Kernel CNN):** Foca na detecção de *n-gramas* e palavras-chave tóxicas (padrões locais), independentemente de sua posição.
4.  **Fusão e Classificação:** As saídas dos dois ramos são concatenadas, oferecendo uma visão rica e diversificada para a camada densa final.

> *Nota: Além de Bi-LSTM, testei o modelo com GRU, mas não houve melhoria de desempenho significativa, optando-se por manter o LSTM.*

---

## Engenharia de Dados e Treinamento

Para garantir a robustez dos resultados:

*   **Dataset:** [Jigsaw/Civil Comments](https://huggingface.co/datasets/civil_comments).
*   **Balanceamento:** Utilização de **Subamostragem (Undersampling)** nas classes majoritárias e **Oversampling** nas minoritárias para garantir um dataset equilibrado.
*   **Validação:** Validação Cruzada Estratificada (*Stratified K-Fold*) de 5 folds.
*   **Otimização:** Uso de *Early Stopping* para prevenir *overfitting*.

---

## Resultados Atuais

| Métrica | Resultado Médio | Meta |
| :--- | :--- | :--- |
| **F1-Score** | **0.90** | 0.93 |

> *Nota: O resultado de 0.90 é robusto, mas notei um problema em uma das classes do dataset; possivelmente isto está afetando os resultados, que poderiam ser maiores. Contudo, meu projeto continua ativo com o objetivo de refinar a arquitetura para alcançar no mínimo 0.93, além de validar o modelo em datasets em Português-BR (o que acredito que será facilitado pela não utilização de embeddings pré-treinados).*

---

## Referências Bibliográficas

Este projeto foi fundamentado nas seguintes obras:

1.  **PITSILIS, G. K.** *Improved two-stage hate speech classification for twitter based on Deep Neural Networks*. arXiv preprint arXiv:2206.04162, 2022.
2.  **ZHOU, C. et al.** *A C-LSTM Neural Network for Text Classification*. COLING 2016.
3.  **SCHUSTER, M.; PALIWAL, K. K.** *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 1997.
4.  **JIGSAW/GOOGLE.** *Jigsaw Unintended Bias in Toxicity Classification*. Kaggle, 2019.

---

## Como Executar

### Requisitos
*   Python 3.8+
*   TensorFlow 2.x
*   Bibliotecas listadas em `requirements.txt`

### Instalação
```bash
git clone https://github.com/Umbura/Hatespeech_Detection_Civil_Comments_NLP.git
pip install -r requirements.txt
```
### Licença
Distribuído sob a licença Apache 2.0. Veja o arquivo LICENSE para mais detalhes.
