import random
import numpy as np
import math


class NeuralNetwork:
    def __init__(self,
                 sizes):  # sizes=[11,3,11] - ilosc neuronow w bazie np. 11 w wejsciowej, bo tyle mam cech, n w ukrytej, 11 na wyjsciu, bo skala jest 0-10 jakosci wina
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.rand(x, y) for x, y in
                        zip(sizes[:-1], sizes[1:])]  # for tworzy nam macierz z naszymi warstwami

    def feedforward(self, sample):

        # print(self.weights)
        for w in self.weights:
            sample = activationFunction1(np.dot(sample, w))
        # print(sample)
        # print(sample)
        return sample  # wynik

    def setWeights(self,
                   weightsFromHeuristic):  # pobiera wektor wag(np. poprzez heurystyke) i wpisuje go do self.weights
        self.weights = weightsFromHeuristic

    def error(self, sample):  # sample[at1,at2,at3,at4,0,0,1]; lub sample[at1,at2,at3,at4], y=[0,0,1]
        # założenie
        #  jakosci wina dostepne:
        # [1,0,0,0,0,0,0,0,0,0,0] - 10
        # [0,1,0,0,0,0,0,0,0,0,0] - 9
        # ...
        # [0,0,0,0,0,0,0,0,0,0,1] - 0

        # obliczenie po wartości sieci (korzystamy z feedforward())

        e = self.feedforward(sample[0:-1])

        slownik_jakosci_wina = {
            0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            2: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            3: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            4: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            7: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            8: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            9: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            10: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        }

        # obliczamy wartość błędu średniokwadratowego z wyniku sieci i wartości oczekiwanej
        suma = 0

        for index_tablicy_errora in range(0, len(e)):
            suma = (e[index_tablicy_errora] - slownik_jakosci_wina[sample[-1]][index_tablicy_errora]) ** 2 + suma

        suma = suma * (1 / 2)

        # zwrócenie błędu dla pojedynczej próbki
        # print(suma)

        return suma


# Projekt powinien analizować dokładność klasyfikatora ze względu na różną ilość warstw ukrytych, ilość neuronów

# różne funkcje aktywacyjne
def activationFunction1(x):  # tanh
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def activationFunction2(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid function unipolarna


def activationFunction3(x):
    return (2.0 / (1.0 + np.exp(-x))) - 1  # sigmoid funkcja bipolarna


def globalErrorOfNeuralNetwork(weights, trainSet, sizes):
    # stworzenie sieci neuronowej
    net = NeuralNetwork(sizes)

    # ustawienie wag z weights do siecie dzieki setWeights()
    net.setWeights(weights)

    errors = []
    # sprawdzamy sie sprawuja wyszkolone wagi(ta macierz)
    # a w tym miejscu sprawdza na ile sie szykolił
    # w pętli dla każdego rekordu w trainSet
    for sample in trainSet:
        wynik_errora = net.error(sample)
        errors.append(wynik_errora)
        # obliczamy błąd sdla sieci dzięki error()
        # error() wyjdzie jakas wartosc
        # ta wartosc wrzucamy do errors
        # błąd dodajemy do listy błędów errors
    # obliczenie błędu całej sieci:
    return sum(errors) / len(errors)


def losuj_wagi(temperatura, sizes):
    # zwraca wagi
    return [np.random.rand(x, y) for x, y in zip(sizes[:-1], sizes[1:])]


# PSO lub inna heurystyka:
def wyzarzanie(liczba_iteracji_w_epoce, temperatura_poczatkowa, wspolczynik_zmiany, temperatura_koncowa, sizes,
               trainSet):
    # przygotowujemy tablice na wagi
    weights = []
    # losowanie wstepnych wag do tablicy
    for i in range(0, liczba_iteracji_w_epoce):
        weights.append([np.random.rand(x, y) for x, y in zip(sizes[:-1], sizes[1:])])

    # heurystyka -> algorytm symulowanego wyżarzania
    # dopóki nasza temperatura_poczatkowa nie będzie mniejsza od koncowej wykonuj
    while (temperatura_poczatkowa > temperatura_koncowa):
        # iteracja
        #print("Temperatura:", temperatura_poczatkowa)
        for i in range(0, liczba_iteracji_w_epoce):

            # tu losujemy sobie nowa przykladowa wage, która zależna jest od temperatury
            nowe_wagi = losuj_wagi(temperatura_poczatkowa, sizes)

            # jeżeli aktualnie porównywana waga(nowa, zależna od  temp) jest lepsza od wcześniej wylosowanych to ją zachowaj
            if globalErrorOfNeuralNetwork(weights[i], trainSet, sizes) > globalErrorOfNeuralNetwork(nowe_wagi, trainSet,
                                                                                                    sizes):
                # jezeli tak to je zachowaj
                weights[i] = nowe_wagi
            # w przeciwnym bądź razie ...
            else:
                # wyznacz B
                B = random.uniform(0, 1)  # losowanie liczby z przedzialu od 0 do 1
                # wylicznie za pomoca wzoru wartosci
                exp = math.exp((globalErrorOfNeuralNetwork(weights[i], trainSet, sizes) - globalErrorOfNeuralNetwork(
                    nowe_wagi, trainSet, sizes)) / temperatura_poczatkowa)
                # porownanie B z exp, jezeli wartosc wylosowana jest mniejsza od
                # obliczonego prawdopodobienstaw wowczas przyjmujemy nowe rozwiazanie(mimo iz jest gorsze)
                if B < exp:
                    weights[i] = nowe_wagi
            # wyznaczenie nowej temperatury
        temperatura_poczatkowa = temperatura_poczatkowa * wspolczynik_zmiany

    # wyznaczenie poprawności dla pierwszego zbioru wag
    sum_error = globalErrorOfNeuralNetwork(weights[0], trainSet, sizes)
    # założenie, że pierwszy jest najlepszy
    bestweights = weights[0]
    # szukanie najlepszego rozwiązania
    for i in range(1, liczba_iteracji_w_epoce):

        r = globalErrorOfNeuralNetwork(weights[i], trainSet, sizes)
        if (r < sum_error):
            sum_error = r
            bestweights = weights[i]

    return bestweights


def make_decision(validatingSet, trainSet, sizes):
    # tworzenie sieci
    net = NeuralNetwork(sizes)

    # wyzarzanie(liczba_iteracji_w_epoce,temperatura_poczatkowa,wspolczynik_zmiany,temperatura_koncowa, sizes,trainSet)
    weights = wyzarzanie(10, 0.95, 0.89, 0.60, sizes, trainSet)

    # ustawienie nowej wagi(wytrenowane)
    net.setWeights(weights)

    corrected = 0
    for sample in validatingSet:

        # wysyłamy naszego sampla do naszej wyuczonej sieci i sprawdzamy, jaka decyzje nam zwraca
        decision = net.feedforward(sample[0:-1])

        index_maxa = decision.tolist().index(max(decision))

        # zapisujemy wynik naszego werdyktu, czyli np. 2 index to virginica
        # print(int(sample[-1]),"<-- sample, index_max -->",index_maxa)
        # sprawdzamy czy wynik jest taki sam, jak to co jest zapisane w pliku
        if (int(sample[-1]) == index_maxa):
            # jeżeli poprawne to zlicz
            corrected += 1

    print("Ilość dobrych trafień " + str(corrected) + " na wszyskie " + str(len(validatingSet)))
    accuracy = str(corrected / len(validatingSet) * 100).split(".")[0] + "% poprawności naszej sieci neuronowej.\nSprawdzanie dla funkcji aktywacyjnej tanh. 3 warstwy ukryte z 9 neuronami."

    return accuracy


# funkcja mieszajaca
def shuffle(X):
    for i in range(0, len(X), 1):
        j = random.randint(0, len(X) - 1)
        X[i], X[j] = X[j], X[i]


# import danych z jakosciawina
plik = open("winequality-white.csv")
tablica_danych = []
plik.readline()
for linia in plik:
    linia = linia.split(";")
    linia[-1] = linia[-1][:-1]
    for i in range(0, len(linia)):
        linia[i] = float(linia[i])
    tablica_danych.append(linia)

# mieszanie
shuffle(tablica_danych)
# wyznaczanie dwóch tablic(treningowej i walidacyjnej)
dl = len(tablica_danych)
dl_trening = int(dl * 0.7);
# rozdzielenie na dwie tablice
tablica_treningowa = tablica_danych[0:dl_trening]
tablica_walidacyjna = tablica_danych[dl_trening:dl + 1]

sizes = [11, 9,9,9, 11]  # LICZBA WARSTW I NEURONÓW W WARSTWACH
#edytowac te srodkowe, ilosc to warstwy ukryte, numer to ilosc neuronów

# wywołanie funkcji make_decision
print(make_decision(tablica_walidacyjna, tablica_treningowa, sizes), "\nWynik nr 1\n\n")
print(make_decision(tablica_walidacyjna, tablica_treningowa, sizes), "\nWynik nr 2\n\n")
print(make_decision(tablica_walidacyjna, tablica_treningowa, sizes), "\nWynik nr 3\n\n")
print(make_decision(tablica_walidacyjna, tablica_treningowa, sizes), "\nWynik nr 4\n\n")
# SPRAWDZAMY DLA FUNKCJI AKCTYWACYJNEJ: sigmoid - bipolarna