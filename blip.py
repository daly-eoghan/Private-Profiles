import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bloomFilter import BloomFilter
import random
from faker import Faker
import copy

def print_bloom_filter(bloom):

    print("[" + ", ".join([str(bit) for bit in bloom.bloomFilter]) + "]")

def bit_flipping(bloom, p = 0.5):

    for x in range(len(bloom.bloomFilter)):
        # random -> float value between 0 and 1.
        if random.random() <= p:
            # Apply an XOR function to the bit value 
            bloom.bloomFilter[x] = bloom.bloomFilter[x] ^ 1

    return bloom

def create_bloom_dicts(bloom_size, number_of_hashes, profile_array, bit_flip_possibility, bloom_and_blip):
    # Create the bloom filter for each person in the dataset 
    # and add all of the profile items.
    bloom_dict = {}
    bloom_dict_flipped = {}

    for row in profile_array:
        bloom = BloomFilter(bloom_size, number_of_hashes)

        for index in range(1, len(row)):
            bloom.add(row[index])
        
        bloom_dict[row[0]] = bloom

        # Avoid having to add all the items into another bloom filter object.
        new_bloom = copy.deepcopy(bloom)
 
        # Bit flip here to create the perturbed bloom filter and save it in the correct 
        # dictionary. 
        bloom_dict_flipped[row[0]] = bit_flipping(new_bloom, bit_flip_possibility)

    # Remove the names from the array so you have only the items.
    profile_array = np.delete(profile_array, 0, 1)

    if bloom_and_blip:
        return bloom_dict, bloom_dict_flipped, profile_array
    
    return bloom_dict_flipped, profile_array

def test_bloom_filter(data, number_of_hashes):
    # Testing Bloom filter functionality.

    # Takes as arguments the size of the bloom filter and the 
    # number of hash functions used. 
    b1 = BloomFilter(len(data), number_of_hashes)

    for x in data:
        b1.add(x)
    
    b1.generateStats()
    print_bloom_filter(b1)

def test_bit_flipping(bloom, p):
    # Shows that bits are flipped.

    print("\nTesting to see if bits are flipped according to the p value.\n")
    print("Original:")
    print_bloom_filter(bloom)
    bit_flipping(bloom, p)
    print("\nResult:")
    print_bloom_filter(bloom)

def test_bloom_filter_content(bloom_data):
    """
    Need to distinguish whether the original elements are present or not. 
    Lets traverse through the bloom filters in the dictionary and use the contains
    method on each item from the corresponding list of the profile array.
    """
    bloom_dict = bloom_data[0]
    profile_array = bloom_data[1]

    blooms = list(bloom_dict.values())
    number_items_not_present = 0
    for row in range(len(profile_array)):
        for index in range(len(profile_array[row])):
            if not blooms[row].contains(profile_array[row][index]):
                number_items_not_present += 1
            # print(profile_array[row][index] + " is present: " + str(blooms[row].contains(profile_array[row][index])))
    
    return (number_items_not_present / (len(profile_array) * len(profile_array[0]))) * 100

def faker_experiment_blip(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility = 0.5, bloom_and_blip = False):

    fake = Faker()
    fake.seed_instance(4321)

    df = pd.DataFrame([ {'name': fake.name(), 'company': fake.company(), 'job': fake.job(), 'country': fake.country()} for _ in range(dataset_size)])

    profile_array = df.to_numpy()

    return create_bloom_dicts(bloom_size, number_of_hashes, profile_array, bit_flip_possibility, bloom_and_blip)

def zoo_experiment_blip(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility = 0.5, bloom_and_blip = False):
    
    df = pd.read_csv('zooData/zoo.csv')

    # Need to shuffle the dataset so the size of the dataset thats used
    # is not used as a clue to the set of animals included.
    df = df.sample(frac=1).reset_index(drop=True)

    if (dataset_size > 0) and (dataset_size < len(df)):
        df = df.iloc[:dataset_size]

    # Change the dataset -> instead of having binary values to respresent 
    # presence add the name of the column as a prefix to reduce the number 
    # of collisions in the bloom filter.
    for col in df.columns[1:]:
        df[col] = col + df[col].astype(str)
    
    profile_array = df.to_numpy()

    return create_bloom_dicts(bloom_size, number_of_hashes, profile_array, bit_flip_possibility, bloom_and_blip)

def run_experiments(bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs, experiment_name):

    experiment_dict = {
        'faker': faker_experiment_blip,
        'zoo': zoo_experiment_blip
    }
    # If the inputted experiment isn't in the dictionary then use Zoo 
    # as a default.
    if experiment_name not in experiment_dict.keys():
        experiment_name = 'zoo'

    # Want to run the experiment with different flip possibilities
    # If hash_values and bit_flip_values are a single integer, convert to a list.
    if type(hash_values) != list:
        hash_values = [hash_values]

    if type(bit_flip_values) != list:
        bit_flip_values = [bit_flip_values]

    # Compare results when different no. of hash functions are used.
    hash_results = []
    for hash in range(len(hash_values)):
        
        results = []
        for _ in range(number_of_runs):

          results.append(test_bloom_filter_content(experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])))

        hash_results.append(results)

    flip_results = []
    bloom_data = experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[0], bit_flip_values[0], True)
    bloom_dict, profile_array =  bloom_data[0], bloom_data[2]

     # Compare results when different bit-flip probabilities are used.
    for flip in range(len(bit_flip_values)):
        
        results = []
        bloom_list = list(bloom_dict.values())
        for _ in range(number_of_runs):
        
            for bloom in range(len(bloom_list)):
                new_bloom = bit_flipping(bloom_list[bloom], bit_flip_values[flip])
                bloom_dict[bloom] = new_bloom

            results.append(test_bloom_filter_content((bloom_dict, profile_array)))

        flip_results.append(results)

    hash_mean = np.mean(hash_results, axis = 1)
    flip_mean = np.mean(flip_results, axis = 1)

    # Create graphs to display the change in percentage of items not found
    # when changing the hash functions/flipping possibilities.
    plt.title(experiment_name.capitalize() + " Hash Experimentation")
    plt.plot(hash_values, hash_mean, label = "Hash Results")
    plt.xlabel("No. of Hash Functions Used")
    plt.ylabel("Percentage of Items not Found")
    plt.show()

    plt.title( experiment_name.capitalize() + " Flip Experimentation")
    plt.plot(bit_flip_values, flip_mean, label = "Flip Results")
    plt.xlabel("Probability of Bit flip Used")
    plt.ylabel("Percentage of Items not Found")
    plt.show()

if __name__ == "__main__":

    # states = """
    #     Alabama Alaska Arizona Arkansas California Colorado Connecticut
    #     Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas
    #     Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota
    #     Mississippi Missouri Montana Nebraska Nevada NewHampshire NewJersey
    #     NewMexico NewYork NorthCarolina NorthDakota Ohio Oklahoma Oregon
    #     Pennsylvania RhodeIsland SouthCarolina SouthDakota Tennessee Texas Utah
    #     Vermont Virginia Washington WestVirginia Wisconsin Wyoming
    #     """.split()

    # # Params: data, number_of_hashes
    # test_bloom_filter(states, 2)

    # # Testing Bit flipping functionality.
    # b1 = BloomFilter(len(states), 2)

    # # Try flipping no bits
    # # Params: bloom, p
    # test_bit_flipping(b1, 0)

    # # Try flipping all bits
    # # Params: bloom, p
    # test_bit_flipping(b1, 1)

    # # First Experiment using the Faker dataset.
    # # Params: bloom_size, dataset_size, number_of_hashes, bit_flip_possibility
    # print(test_bloom_filter_content(faker_experiment_blip(30, 10, 2, 0.5)))

    # # Params: bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs
    # run_experiments(50, 10, list(range(2, 8)), [0.2, 0.5, 0.8], 10, "faker")
    # run_experiments(100, 50, list(range(21, 26)), [0.25, 0.5, 0.75, 0.9], 10, "faker")

    # # First Experiment using the Zoo animal dataset.
    # # Params: bloom_size, dataset_size, number_of_hashes, bit_flip_possibility.
    # print(test_bloom_filter_content(zoo_experiment_blip(30, 10, 2, 0.5)))
    
    # # Params: bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs
    run_experiments(50, 10, list(range(2, 8)), [0.2, 0.5, 0.8], 10, "zoo")
    run_experiments(100, 50, list(range(21, 26)), [0.25, 0.5, 0.75, 0.9], 10, "zoo")