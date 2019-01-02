import numpy as np # para processar dados matemáticos e cálculo
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
import sys # para fazer os inputs do utilizador
import matplotlib.pyplot as plt
from keras.models import load_model

ficheiro_csv_treino = 'pima-indians-diabetes_normalizado.csv'
ficheiro_csv_teste = 'pima-indians-diabetes_teste_v1_random_normalizado.csv'
ficheiro_model_json = 'model.json'
ficheiro_weights_h5 = 'weights.h5'

EPOCH_NUMBER = 10000
BATCH_SIZE = 10

def treinar_rede(ficheiro_csv_treino, ficheiro_model_json, ficheiro_weights_h5, epoch_number, batch_size):
 
    # carrega os dados do ficheiro CSV para um array global "dataset"
    dataset_treino = np.loadtxt(ficheiro_csv_treino, delimiter=",")

    # Carrega os parametros de entrada (X) e os de saída (Y) com o array "dataset_treino"
    X_treino = dataset_treino[:, 0:8]  #
    Y_treino = dataset_treino[:, 8]  #

    # parametriza o modelo neural, indicando que a parametrização é instruída de forma sequencial, os neurónios estão densamente ligados (todos com todos)
    # 5 camadas, 1 de entrada (15), 3 intermédias (10, 8 e 10), 1 de saída (1)
    model = Sequential()
    model.add(Dense(15, input_dim=8, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # compila a rede e indica o modelo de estimação do erro, optimização do erro e método/mérito de mensuração do erro.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # define os arrays de entrada e de saída, o número de iterações de treino, o número de entradas por iteração de treino (batch) e mescla as entradas (shuffle)
    model_fit_values = model.fit(X_treino, Y_treino, epochs=epoch_number, batch_size=batch_size, shuffle='true')

    # treina o modelo e devolve os scores para print posterior
    scores = model.evaluate(X_treino, Y_treino)

    # mostra o nível de precisão que o treino atingiu
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # guarda em json o modelo neural
    model_json = model.to_json()
    with open(ficheiro_model_json, "w") as json_file:
        json_file.write(model_json)

    # guarda em HDF5 a parametrização treinada da rede neural
    model.save(ficheiro_weights_h5)
    
    accuracy = model_fit_values.history['acc']
    loss = model_fit_values.history['loss']
    epochs_size = range(len(accuracy))
    plt.plot(epochs_size, accuracy, 'bo', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs_size, loss, 'bo', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()
    
    return scores

def carregar_rede_neural(ficheiro_model_json, ficheiro_weights_h5):
    # carrega a estrutura da rede neural
    json_file = open(ficheiro_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # carrega os parametros da rede neural
    loaded_model.load_weights(ficheiro_weights_h5)

    return loaded_model

if input("Deseja treinar a rede neural? (y/n): ") == "y":
    print("\nA iniciar o treino...")
    treinar_rede(ficheiro_csv_treino, ficheiro_model_json, ficheiro_weights_h5, EPOCH_NUMBER, BATCH_SIZE)
    print("Treino concluído.")

if input("Deseja testar a rede neural? (y/n): ") == "y":
    print("\nA iniciar o carregamemento da rede e parâmetros...")
    loaded_model = carregar_rede_neural(ficheiro_model_json, ficheiro_weights_h5)
    print("Carregamento concluído.\n")
    
    # carrega os dados do ficheiro CSV para um array global "dataset"
    dataset_teste = np.loadtxt(ficheiro_csv_teste, delimiter=",")
    
    X_teste = dataset_teste[:, 0:8]  #
    Y_teste = loaded_model.predict_classes(X_teste)
    print(X_teste)
    print(Y_teste)
