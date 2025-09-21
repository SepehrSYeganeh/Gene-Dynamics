from genedyn import *


def main():
    number_of_stable_fixed_points = 512
    generate_data(number_of_stable_fixed_points)
    cluster_stable_points()
    clustering_fixed_points()
    learning_dynamics_custom_sampling()
    pass


if __name__ == '__main__':
    main()
