import profile
import pandas as pd
import numpy as np
from bloomFilter import BloomFilter
import random
from faker import Faker
import copy

def print_bloom_filter(bloom):

    print("[" + ", ".join([str(bit) for bit in bloom.bloomFilter]) + "]")

def bit_flipping(bloom, p = 0.5):

    for x in range(len(bloom.bloomFilter)):
        # random -> float value between 0 and 1.
        if random.random() > p:
            # Apply an XOR function to the bit value 
            bloom.bloomFilter[x] = bloom.bloomFilter[x] ^ 1

    return bloom

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
    # Shows that bits are flipped according to the p value.

    print("\nTesting to see if bits are flipped according to the p value.\n")
    print("Original:")
    print_bloom_filter(bloom)
    bit_flipping(bloom, p)
    print("\nResult:")
    print_bloom_filter(bloom)

def test_bloom_filter_content(bloom_dict, profile_array):
    """
    Need to distinguish whether the original elements are present or not. 
    Lets traverse through the bloom filters in the dictionary and use the contains
    method on each item from the corresponding list of the profile array.
    """
    blooms = list(bloom_dict.values())
    number_items_not_present = 0
    for row in range(len(profile_array)):
        for index in range(len(profile_array[row])):
            if not blooms[row].contains(profile_array[row][index]):
                number_items_not_present += 1
            # print(profile_array[row][index] + " is present: " + str(blooms[row].contains(profile_array[row][index])))
    
    return number_items_not_present / (len(profile_array) * len(profile_array[0]))

def faker_experiment_blip(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility):

    fake = Faker()
    fake.seed_instance(4321)

    df = pd.DataFrame([ {'name': fake.name(), 'company': fake.company(), 'job': fake.job(), 'country': fake.country()} for _ in range(dataset_size)])

    profile_array = df.to_numpy()
    bloom_dict = {}
    bloom_dict_flipped = {}
    
    # Create the bloom filter for each person in the dataset 
    # and add all of the profile items.
    for row in profile_array:
        new_bloom = BloomFilter(bloom_size, number_of_hashes)

        for index in range(1, len(row)):
            new_bloom.add(row[index])
        
        bloom_dict[row[0]] = new_bloom

        # Avoid having to add all the items into another bloom filter object.
        new_flipped_bloom = copy.deepcopy(new_bloom)
 
        # Bit flip here to create the perturbed bloom filter and save it in the correct 
        # dictionary. 
        bloom_dict_flipped[row[0]] = bit_flipping(new_flipped_bloom, bit_flip_possibility)

    # Remove the names from the array so you have only the items.
    profile_array = np.delete(profile_array, 0, 1)

    # Check the percentage of the original items that could not be identified in each scenario.
    return test_bloom_filter_content(bloom_dict, profile_array) * 100, test_bloom_filter_content(bloom_dict_flipped, profile_array) * 100

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

    # # Try flipping all bits
    # # Params: bloom, p
    # test_bit_flipping(b1, 0)

    # # Try flipping no bits
    # # Params: bloom, p
    # test_bit_flipping(b1, 1)

    # First Experiment using the Faker dataset.
    # Params: bloom_size, dataset_size, number_of_hashes, bit_flip_possibility
    results = faker_experiment_blip(50, 10, 2, 0.2)
    print("Faker Experiment results -> Without bit flip: {} With bit flip: {}".format(results[0], results[1]))

    