# pima-indians-diabetes
Demonstração de uma rede neural aplicada à deteção de diabetes, com base em dados biométricos.

Trata-se de uma rede neural com a estrutura de 5 camadas: 1 de entrada (15 neurónios), 3 intermédias (10, 8 e 10 neurónios, respectivamente), 1 de saída (1 neurónio). Em todas as camadas foi usada a função de ativação sigmoid(): https://en.wikipedia.org/wiki/Sigmoid_function.

Procedeu-se à normalização dos dados para facilitar a convergência da rede, dado ter-se optado pela função de ativação sigmoid().

Utilizou-se a biblioteca Keras para obter suporte à implementação com o método do TensorFlow. Escrito e testado em Python 3.6, com Keras 2.2.4 e TensorFlow 1.12.0.

O código está comentado para facilitar a leitura e aprendizagem.

Fonte #1: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/.

Fonte #2: https://www.kaggle.com/uciml/pima-indians-diabetes-database.
