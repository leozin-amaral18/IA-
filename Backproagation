import numpy as np
import random
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    # Inicializando os vetores e matrizes
    I0 = np.zeros(4)
    O0 = np.zeros(4)
    I1 = np.zeros(4)
    O1 = np.zeros(4)
    I2 = np.zeros(2)
    O2 = np.zeros(2)
    w1 = np.zeros((4, 4))
    w2 = np.zeros((4, 2))
    nw1 = np.zeros((4, 4))
    nw2 = np.zeros((4, 2))
    vw1 = np.zeros((4, 4))
    vw2 = np.zeros((4, 2))
    d1 = np.zeros(3)

    # Vetores de entrada e saída esperada
    E1 = np.array([0.0, 0.0, 1.0, 1.0])
    E2 = np.array([0.0, 1.0, 0.0, 1.0])
    t = np.array([0.0, 1.0, 1.0, 0.0])

    # Inicialização dos pesos aleatórios (usando índice 0 para compatibilidade)
    w1[0][0] = random.uniform(-2.0, 2.0)
    vw1[0][0] = 0.0
    w1[1][0] = random.uniform(-2.0, 2.0)
    vw1[1][0] = 0.0
    w1[2][0] = random.uniform(-2.0, 2.0)
    vw1[2][0] = 0.0

    w1[0][1] = random.uniform(-2.0, 2.0)
    vw1[0][1] = 0.0
    w1[1][1] = random.uniform(-2.0, 2.0)
    vw1[1][1] = 0.0
    w1[2][1] = random.uniform(-2.0, 2.0)
    vw1[2][1] = 0.0

    w2[0][0] = random.uniform(-2.0, 2.0)
    vw2[0][0] = 0.0
    w2[1][0] = random.uniform(-2.0, 2.0)
    vw2[1][0] = 0.0
    w2[2][0] = random.uniform(-2.0, 2.0)
    vw2[2][0] = 0.0

    # Início do loop de treinamento
    for m in range(1001):
        er = 0.0
        for n in range(1001):
            cs = random.randint(0, 3)

            # Propagação para frente
            I0[1] = E1[cs]
            I0[2] = E2[cs]
            I0[3] = 1.0  # Bias

            O0[1] = I0[1]
            O0[2] = I0[2]
            O0[3] = I0[3]

            I1[1] = O0[1] * w1[0][0] + O0[2] * w1[1][0] + O0[3] * w1[2][0]
            I1[2] = O0[1] * w1[0][1] + O0[2] * w1[1][1] + O0[3] * w1[2][1]
            I1[3] = 1.0  # Bias

            O1[1] = sigmoid(I1[1])
            O1[2] = sigmoid(I1[2])
            O1[3] = I1[3]  # Bias

            I2[1] = O1[1] * w2[0][0] + O1[2] * w2[1][0] + O1[3] * w2[2][0]
            O2[1] = sigmoid(I2[1])

            # Backpropagation
            d2 = (t[cs] - O2[1]) * O2[1] * (1.0 - O2[1])
            d1[1] = O1[1] * (1.0 - O1[1]) * d2 * w2[0][0]
            d1[2] = O1[2] * (1.0 - O1[2]) * d2 * w2[1][0]

            # Atualizando pesos da camada de saída
            nw2[0][0] = w2[0][0] + 0.5 * d2 * O1[1] + 0.5 * vw2[0][0]
            vw2[0][0] = nw2[0][0] - w2[0][0]
            w2[0][0] = nw2[0][0]

            nw2[1][0] = w2[1][0] + 0.5 * d2 * O1[2] + 0.5 * vw2[1][0]
            vw2[1][0] = nw2[1][0] - w2[1][0]
            w2[1][0] = nw2[1][0]

            nw2[2][0] = w2[2][0] + 0.5 * d2 * O1[3] + 0.5 * vw2[2][0]
            vw2[2][0] = nw2[2][0] - w2[2][0]
            w2[2][0] = nw2[2][0]

            # Atualizando pesos da primeira camada oculta
            nw1[0][0] = w1[0][0] + 0.5 * d1[1] * O0[1] + 0.5 * vw1[0][0]
            vw1[0][0] = nw1[0][0] - w1[0][0]
            w1[0][0] = nw1[0][0]

            nw1[1][0] = w1[1][0] + 0.5 * d1[1] * O0[2] + 0.5 * vw1[1][0]
            vw1[1][0] = nw1[1][0] - w1[1][0]
            w1[1][0] = nw1[1][0]

            nw1[2][0] = w1[2][0] + 0.5 * d1[1] * O0[3] + 0.5 * vw1[2][0]
            vw1[2][0] = nw1[2][0] - w1[2][0]
            w1[2][0] = nw1[2][0]

            nw1[0][1] = w1[0][1] + 0.5 * d1[2] * O0[1] + 0.5 * vw1[0][1]
            vw1[0][1] = nw1[0][1] - w1[0][1]
            w1[0][1] = nw1[0][1]

            nw1[1][1] = w1[1][1] + 0.5 * d1[2] * O0[2] + 0.5 * vw1[1][1]
            vw1[1][1] = nw1[1][1] - w1[1][1]
            w1[1][1] = nw1[1][1]

            nw1[2][1] = w1[2][1] + 0.5 * d1[2] * O0[3] + 0.5 * vw1[2][1]
            vw1[2][1] = nw1[2][1] - w1[2][1]
            w1[2][1] = nw1[2][1]

            # Acumula erro
            er += (t[cs] - O2[1]) ** 2

        er /= 1000.0
        print(f"{m} {er}")

    # Exibe pesos finais
    print(f"{w1[0][0]} {w1[1][0]} {w1[2][0]}")
    print(f"{w1[0][1]} {w1[1][1]} {w1[2][1]}")
    print(f"{w1[0][0]} {w1[1][0]} {w1[2][0]}")


if __name__ == "__main__":
    main()
