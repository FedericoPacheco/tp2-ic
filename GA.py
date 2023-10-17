import pygad.kerasga
import tensorflow #Me da error cuando esta tensorflow.keras, prueben asi
import numpy as np
import matplotlib.pyplot as plt

# Optimizar performance (no probado):
# https://www.tensorflow.org/install/source_windows?hl=es-419
# Ver: ROCm, DirectML para usar GPU AMD


# Funcion general para el calculo del fitness de cada individuo
def fitness_func(ga_instance, solution, sol_idx, x_arr, y_true_arr, model):
    
    # Transformar el el cromosoma de arreglo 1D (cromosoma) a matriz y asignar al modelo de keras
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model = model, weights_vector = solution)
    model.set_weights(weights = model_weights_matrix)

    # Predecir con keras y calcular el error
    y_predic_arr = model.predict(x_arr)
    error_class = tensorflow.keras.losses.MeanSquaredError()
    solution_fitness = 1.0 / (error_class(y_true_arr, y_predic_arr).numpy() + 0.00000001)

    return solution_fitness


# Imprimir status de la busqueda
def callback_generation(ga_instance):
    print("Generacion = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def hermite_polynomial():

    # Crear configuracion del paper: 1-15-1, con tansig-tansig-reglin
    def create_keras_model():
        input_layer  = tensorflow.keras.layers.Input(1)
        dense_layer = tensorflow.keras.layers.Dense(15, activation="tanh")
        output_layer = tensorflow.keras.layers.Dense(1, activation="linear")

        model = tensorflow.keras.Sequential()
        model.add(input_layer)
        model.add(dense_layer)
        model.add(output_layer)

        print("Cantidad de parametros del modelo: ", model.count_params())
        
        return model
    
    # Generar datos de entrenamiento y prueba, evaluando el polinomio en distintos valores de x
    def generate_data():
        h_vect = lambda x: 1.1 * (1 - x - 2 * x**2) * np.exp(- x**2 / 2)
        x_train_arr = np.linspace(0, 10, 100)
        x_test_arr = np.linspace(0, 15, 100)

        return x_train_arr, h_vect(x_train_arr), x_test_arr, h_vect(x_test_arr)


    print("----GA para polinomio de hermite----")
    print("Entrenamiento:")

    # Parametros
    # Ver: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
    num_solutions = 50 # individuos
    num_generations = 100 # iteraciones
    num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
    # Tipos:
    # sss: steady-state selection, rws: roulette wheel selection, random,
    # sus: stochastic universal selection, rank, tournament
    parent_selection_type = "sss"
    # Tipos: single_point, two_points, uniform, scattered
    crossover_type = "single_point"
    p_crossover = 0.6
    # Tipos: random, swap, inversion, scramble
    mutation_type = "random"
    p_mutation = 0.01

    # Configurar busqueda por GA
    hermite_model = create_keras_model()
    x_train_arr, h_train_arr, x_test_arr, h_test_arr = generate_data()
    keras_ga = pygad.kerasga.KerasGA(model = hermite_model, num_solutions = num_solutions)
    # A nested list holding the model parameters. This list is updated after each generation.
    initial_population = keras_ga.population_weights # shape = (num_solutions, parameters) -> un array 1D de todos los parametros para cada individuo (solucion)
    
    ga_instance = pygad.GA(
        num_generations = num_generations, 
        num_parents_mating = num_parents_mating, 
        initial_population = initial_population,
        fitness_func = lambda ga_instance, solution, sol_idx: fitness_func(ga_instance, solution, sol_idx, x_train_arr, h_train_arr, hermite_model),
        parent_selection_type = parent_selection_type,
        crossover_type = crossover_type,
        crossover_probability = p_crossover,
        mutation_type = mutation_type,
        mutation_probability = p_mutation,
        on_generation = callback_generation
    )

    # Realizar busqueda y guardar los resultados
    ga_instance.run()
    ga_instance.save("./ga_test")

    # Imprimir evolucion de la func de fitness por generacion
    ga_instance.plot_fitness()

    # Asignar mejores pesos hallados al modelo de keras
    best_solution, _, _ = ga_instance.best_solution()
    best_weights = pygad.kerasga.model_weights_as_matrix(model = hermite_model, weights_vector = best_solution)
    hermite_model.set_weights(weights = best_weights)

    # Terminar de entrenar modelo con keras
    hermite_model.compile(loss = "mse", optimizer = "adam", metrics = ["accuracy"])
    # Ya tiene los pesos del AG
    hermite_model.fit(
        x_train_arr, 
        h_train_arr, 
        epochs = 20, 
        validation_data= (x_test_arr, h_test_arr), 
        verbose = 2
    )
    
    # Calcular el error / loss
    error_class = tensorflow.keras.losses.MeanSquaredError()
    best_sol_h_predic_arr = hermite_model.predict(x_test_arr)
    error = error_class(h_test_arr, best_sol_h_predic_arr).numpy()
    
    # Comparar funcion original y predicha
    plt.title("Datos de test: funcion original y predicha (MSE = " + str(error) + ")")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_test_arr, h_test_arr, color = "blue", label = "Original")
    plt.plot(x_test_arr, best_sol_h_predic_arr, "red", label = "Modelo")
    plt.legend(loc = "upper left")
    plt.show()

if __name__ == "__main__":
    hermite_polynomial()

