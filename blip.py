from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from itertools import chain, combinations, product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bloomFilter import BloomFilter
import random
from faker import Faker
import copy
from time import time


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
    # Create the bloom filter for each subject in the dataset 
    # and add all of the profile items.
    bloom_dict = {}
    bloom_dict_flipped = {}

    for row in profile_array:
        bloom = BloomFilter(bloom_size, number_of_hashes)

        for index in range(1, len(row)):
            if row[index] != None:
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

def test_bloom_filter_content(bloom_data, bloom_and_blip = False):
    """
    Need to distinguish whether the original elements are present or not. 
    Lets traverse through the bloom filters in the dictionary and use the contains
    method on each item from the corresponding list of the profile array.
    """

    bloom_dict = bloom_data[0]

    if bloom_and_blip:
        profile_array = bloom_data[2]

    else:
        profile_array = bloom_data[1]

    blooms = list(bloom_dict.values())

    item_count = []
    for row in range(len(profile_array)):
        number_items = 0
        number_items_not_present = 0

        for index in range(len(profile_array[row])):
            if profile_array[row][index] != None:
                number_items += 1

                if not blooms[row].contains(profile_array[row][index]):
                    number_items_not_present += 1

        item_count.append((number_items_not_present, number_items))

    sum_not_found = sum([x[0] for x in item_count])
    sum_items = sum([x[1] for x in item_count])

    # return (number_items_not_present / (len(profile_array) * len(profile_array[0]))) * 100

    # Returning list of blooms to calculate the false positive.
    if sum_not_found > 0:
        return (sum_not_found / sum_items) * 100
    else:
        return 0.0

def count_false_positives_negatives(bloom_data, bloom_and_blip = False):
    """
    Need to test the bloom filters for true/false positives/negatives.
    Lets traverse through the bloom filters in the dictionary and use the contains
    method on each item from the corresponding list of the profile array to determine
    which it is and classify each as one of the four options.
    """

    bloom_dict = bloom_data[0]

    if bloom_and_blip:
        profile_array = bloom_data[2]

    else:
        profile_array = bloom_data[1]

    blooms = list(bloom_dict.values())

    item_count = []
    for row in range(len(profile_array)):
        number_items = 0
        number_false_positives = 0
        number_false_negatives = 0
        number_true_positives = 0
        number_true_negatives = 0

        for index in range(len(profile_array[row])):
            value = profile_array[row][index]
            if value != None:
                number_items += 1

                final_char = value[len(value) - 1]
                # Positive -> was found.
                if  blooms[row].contains(value):
                
                    # False positive, found but was not added.
                    if final_char == "0":
                        number_false_positives += 1
                    
                    # True positive, found and was added.
                    else:
                        number_true_positives += 1

                # Negative -> was not found.
                else:

                    # True negative, wasn't found and wasn't added.
                    if final_char == "0":
                        number_true_negatives += 1

                    # False negative, wasn't found but was added.
                    else:
                        number_false_negatives += 1

        item_count.append((number_items, number_true_positives, number_true_negatives, number_false_positives, number_false_negatives))

    sum_items = sum([x[0] for x in item_count])
    sum_true_positives = sum([x[1] for x in item_count])
    sum_true_negatives = sum([x[2] for x in item_count])
    sum_false_positives = sum([x[3] for x in item_count])
    sum_false_negatives = sum([x[4] for x in item_count])

    return sum_items, sum_true_positives, sum_true_negatives, sum_false_positives, sum_false_negatives

def profile_reconstruction_attack(bloom, dataset, number_of_hashes = 2, p = 0, string_len = 0):
    """
    Exhaust the bloom filter by querying all possible items of the domain. The probability 
    that a bit is flipped is public information. Output a set of items based on their guess 
    of the profile that's hidden beneath -> compare this and our original to see how 
    close they are -> cosine similarity. Possibility of false positives -> might be 
    impossible to recreate the original 100%.
    """
    data = []
    if dataset == "zoo":
        binary_features = [
                        'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
                        'toothed', 'backbone', 'breathes', 'venomous', 'fins','tail', 'domestic', 'catsize'
                        ]
        non_binary_features = [('legs', 2, 8, 2), ('class_type', 1, 7, 1)]

        # Have feature lists with False values and then a check 
        # for false will leave the algorithm skip this feature.
        for feature in binary_features:
            result = [False]
            for x in range(1, 2):
                result.append(feature + str(x))

            data.append(result)
        
        for feature in non_binary_features:
            result = []
            for x in range(feature[1], feature[2] + 1, feature[3]):
                result.append(feature[0] + str(x))
            
            data.append(result)

        data[-2].insert(0, False)

        # Use the Cartesian Product to get all the possible items in the domain.
        possible_combinations = list(product(*data))
    
    elif dataset == "homogenous":
        # Going to use cartesian product so for strings of length A then there needs to be A
        # alphabets added.
        for x in range(string_len):
            data.append(list(map(chr, range(ord('a'), ord('z')+1))))
            
        # Use the Cartesian Product to get all the possible items in the domain.
        data = list(product(*data))

        # Now you want to try all possible subsets of this list of all possible string combinations
        # of a set length. Memory error for the powerset of all possible string combinations.
        # possible_combinations = list(chain.from_iterable(combinations(data, r) for r in range(len(data)+1)))
        # print(possible_combinations)

    similarity_score = 0
    best_guesses = []
    guess_bloom = BloomFilter(bloom.length(), number_of_hashes)
    for set in possible_combinations:
        guess_bloom.reset()
        for val in set:
            if val != False:
                guess_bloom.add(val)
        
        guess_bloom = bit_flipping(guess_bloom, p)
        result = cosine_similarity(np.array(guess_bloom.bloomFilter).reshape(1, -1), np.array(bloom.bloomFilter).reshape(1, -1))
        
        # Repeatedly compare (cosine similarity) the guess blooms with the original and 
        # keep the one with the closest comparison.
        if result > similarity_score:
            similarity_score = result
            best_guesses.append((set, similarity_score))
    
    return best_guesses[-1]

def bloom_clustering(bloom_data, number_of_clusters):
    """
        Finding its c nearest neighbours.

        Split into two stages: Clustering and Searching.

            Clustering:
            
            Assign an initial random cluster to each profile.

            20 rounds -> exchange the bloom of node i with its current neighbours and other
            random nodes -> keep those with high similarity. At the end each node will have 
            its c most similar nodes as its neighbours.

            Searching:

            Check if each node can find the items contained in its search set in the profiles 
            of its neighbours. From this you calculate the mean recall of all nodes. Plot
            results against the level of differential privacy (found in the paper).

    """
    bloom_list = list(bloom_data[0].values())
    profile_array = bloom_data[1]

    bloom_clusters = {}


    # Assign an initial random cluster to each profile.
    for bloom in bloom_list:
        bloom_clusters[bloom] = [random.randint(0, number_of_clusters), []]
    

    for cluster in range(number_of_clusters):
        nodes = [bloom for bloom in bloom_clusters if bloom_clusters[bloom][0] == cluster]
        
    print(bloom_clusters)

def faker_experiment_blip(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility = 0.5, bloom_and_blip = False):

    fake = Faker()
    fake.seed_instance(4321)

    df = pd.DataFrame([ {'name': fake.name(), 'company': fake.company(), 'job': fake.job(), 'country': fake.country()} for _ in range(dataset_size)])

    profile_array = df.to_numpy()

    return create_bloom_dicts(bloom_size, number_of_hashes, profile_array, bit_flip_possibility, bloom_and_blip)

def homogenous_experiment_blip(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility = 0.5, bloom_and_blip = False):

    no_columns = 4
    string_len = 5

    def create_random_string(string_len):

        h_string = ""
        alphabet = list(map(chr, range(ord('a'), ord('z')+1)))
        for _ in range(string_len):
            h_string += alphabet[random.randint(0, 25)]
        return h_string

    dataset = []
    for _ in range(dataset_size):
        row = []
        for _ in range(no_columns):
            row.append(create_random_string(string_len))

        dataset.append(row)

    profile_array = np.array(dataset)

    return create_bloom_dicts(bloom_size, number_of_hashes, profile_array, bit_flip_possibility, bloom_and_blip)

def zoo_experiment_blip(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility = 0.5, bloom_and_blip = False):
    
    df = pd.read_csv('zooData/zoo.csv')

    # Need to shuffle the dataset so the size of the dataset thats used
    # is not used as a clue to the set of animals included.
    # df = df.sample(frac=1).reset_index(drop=True)

    if (dataset_size > 0) and (dataset_size < len(df)):
        df = df.iloc[:dataset_size]

    # Change the dataset -> instead of having binary values to respresent 
    # presence add the name of the column as a prefix to reduce the number 
    # of collisions in the bloom filter. Only addded to the features that are present;
    # values of 0 should not be included.
    
    temp_matrix = []
    for row in df.values.tolist():
        new_entry = [row[0]]
        for value in range(1, len(row)):
            if row[value] > 0:
                row[value] = df.columns[value] + str(row[value])
                new_entry.append(row[value])
        temp_matrix.append(new_entry)
    
    # Includes None values as np arrays are of fixed size.
    profile_array = np.array(pd.DataFrame(temp_matrix))

    return create_bloom_dicts(bloom_size, number_of_hashes, profile_array, bit_flip_possibility, bloom_and_blip)

def zoo_experiment_blip_all_features(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility = 0.5, bloom_and_blip = False):
    
    df = pd.read_csv('zooData/zoo.csv')

    # Need to shuffle the dataset so the size of the dataset thats used
    # is not used as a clue to the set of animals included.
    # df = df.sample(frac=1).reset_index(drop=True)

    if (dataset_size > 0) and (dataset_size < len(df)):
        df = df.iloc[:dataset_size]

    # Change the dataset -> instead of having binary values to respresent 
    # presence add the name of the column as a prefix to reduce the number 
    # of collisions in the bloom filter. Only addded to the features that are present;
    # values of 0 should not be included.
    
    temp_matrix = []
    for row in df.values.tolist():
        new_entry = [row[0]]
        for value in range(1, len(row)):
            row[value] = df.columns[value] + str(row[value])
            new_entry.append(row[value])
        temp_matrix.append(new_entry)
    
    # Includes None values as np arrays are of fixed size.
    profile_array = np.array(pd.DataFrame(temp_matrix))

    return create_bloom_dicts(bloom_size, number_of_hashes, profile_array, bit_flip_possibility, bloom_and_blip)

def run_experiments(bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs, experiment_name):

    experiment_dict = {
        'faker': faker_experiment_blip,
        'homogenous': homogenous_experiment_blip,
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
    hash_results, positives_and_negatives_hash_results, false_values_hash = ([] for i in range(3))
    for hash in range(len(hash_values)):
        
        results = []
        positives_and_negatives = []
        false_values = []
        for _ in range(number_of_runs):
            bloom_data = experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])
            results.append(test_bloom_filter_content(bloom_data))

            # Uses the bloom filter with the correct features and compares it to the 
            # profile_array of all features.
            bloom_data_all_features = zoo_experiment_blip_all_features(bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])
            tuple = count_false_positives_negatives((bloom_data[0], bloom_data_all_features[1]))
            positives_and_negatives.append(tuple)
            false_values.append(tuple[3] + tuple[4])

        hash_results.append(results)
        positives_and_negatives_hash_results.append(positives_and_negatives)
        false_values_hash.append(false_values)

    print(hash_results)
    flip_results, positives_and_negatives_flip_results, false_values_flip = ([] for i in range(3))
    bloom_data = experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[0], bit_flip_values[0], True)
    bloom_data_all_features = zoo_experiment_blip_all_features(bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])
    bloom_dict, profile_array =  bloom_data[0], bloom_data[2]

    # Compare results when different bit-flip probabilities are used.
    for flip in range(len(bit_flip_values)):
        
        results = []
        positives_and_negatives = []
        false_values = []
        bloom_list = list(bloom_dict.values())
        for _ in range(number_of_runs):
        
            for bloom in range(len(bloom_list)):
                new_bloom = bit_flipping(bloom_list[bloom], bit_flip_values[flip])
                bloom_dict[bloom] = new_bloom

            results.append(test_bloom_filter_content((bloom_dict, profile_array)))
            # Pass in the dictionary of newly flipped bloom filters and the profile_array of all features.
            tuple = count_false_positives_negatives((bloom_dict, bloom_data_all_features[1]))
            positives_and_negatives.append(tuple)
            false_values.append(tuple[3] + tuple[4])
            
        flip_results.append(results)
        positives_and_negatives_flip_results.append(positives_and_negatives)
        false_values_flip.append(false_values)

    print(flip_results)
    hash_mean = np.mean(hash_results, axis = 1)
    flip_mean = np.mean(flip_results, axis = 1)
    false_values_hash_mean = np.mean(false_values_hash, axis = 1)
    false_values_flip_mean = np.mean(false_values_flip, axis = 1)
    positives_and_negatives_hash_mean = np.mean(positives_and_negatives_hash_results, axis = 1)
    positives_and_negatives_flip_mean = np.mean(positives_and_negatives_flip_results, axis = 1)
    # print(positives_and_negatives_flip_results)
    # print(positives_and_negatives_flip_mean)

    # Create graphs to display the change in percentage of items not found
    # when changing the hash functions/flipping possibilities.
    # plt.title(experiment_name.capitalize() + " Hash Experimentation")
    # plt.plot(hash_values, hash_mean, label = "Hash Results")
    # plt.xlabel("No. of Hash Functions Used")
    # plt.ylabel("Percentage of Items not Found")
    # plt.show()

    # plt.title(experiment_name.capitalize() + " Flip Experimentation")
    # plt.plot(bit_flip_values, flip_mean, label = "Flip Results")
    # plt.xlabel("Probability of Bit flip Used")
    # plt.ylabel("Percentage of Items not Found")
    # plt.show()

    plt.title(experiment_name.capitalize() + " Positive/Negative Hash Experimentation")
    plt.plot(hash_values, positives_and_negatives_hash_mean, label = ("Total Items", "Total true_positives", "Total true_negatives", "Total false_positives", "Total false_negatives"))
    plt.plot(hash_values, false_values_hash_mean, label = "Sum of False Positives and Negatives")
    plt.xlabel("No. of Hash Functions (k)")
    plt.ylabel("No. of Items")
    plt.legend()
    plt.show()

    plt.title(experiment_name.capitalize() + " Positive/Negative Flip Experimentation")
    plt.plot(bit_flip_values, positives_and_negatives_flip_mean, label = ("Total Items", "Total true_positives", "Total true_negatives", "Total false_positives", "Total false_negatives"))
    plt.plot(bit_flip_values, false_values_flip_mean, label = "Sum of False Positives and Negatives")
    plt.xlabel("Probability of Bit flip Used")
    plt.ylabel("No. of Items")
    plt.legend()
    plt.show()

def run_experiments_all_features(bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs, experiment_name):

    experiment_dict = {
        # 'faker': faker_experiment_blip,
        # 'homogenous': homogenous_experiment_blip,
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
    hash_results_all_features = []
    for hash in range(len(hash_values)):
        
        results = []
        results_all_features = []
        for _ in range(number_of_runs):

            results.append(test_bloom_filter_content(experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])))
            results_all_features.append(test_bloom_filter_content(zoo_experiment_blip_all_features(bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])))


        hash_results.append(results)
        hash_results_all_features.append(results_all_features)

    flip_results = []
    bloom_data = experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[0], bit_flip_values[0], True)
    bloom_dict, profile_array =  bloom_data[0], bloom_data[2]

    flip_results_all_features = []
    bloom_data_all_features = zoo_experiment_blip_all_features(bloom_size, dataset_size, hash_values[0], bit_flip_values[0], True)
    bloom_dict_all_features, profile_array_all_features =  bloom_data_all_features[0], bloom_data_all_features[2]

     # Compare results when different bit-flip probabilities are used.
    for flip in range(len(bit_flip_values)):
        
        results = []
        bloom_list = list(bloom_dict.values())

        results_all_features = []
        bloom_list_all_features = list(bloom_dict_all_features.values())
        for _ in range(number_of_runs):
        
            for bloom in range(len(bloom_list)):
                new_bloom = bit_flipping(bloom_list[bloom], bit_flip_values[flip])
                bloom_dict[bloom] = new_bloom

            results.append(test_bloom_filter_content((bloom_dict, profile_array)))
            
            for bloom in range(len(bloom_list_all_features)):
                new_bloom = bit_flipping(bloom_list_all_features[bloom], bit_flip_values[flip])
                bloom_dict_all_features[bloom] = new_bloom

            results_all_features.append(test_bloom_filter_content((bloom_dict_all_features, profile_array_all_features)))

        flip_results.append(results)
        flip_results_all_features.append(results_all_features)


    hash_mean = np.mean(hash_results, axis = 1)
    flip_mean = np.mean(flip_results, axis = 1)

    hash_mean_all_features = np.mean(hash_results_all_features, axis = 1)
    flip_mean_all_features = np.mean(flip_results_all_features, axis = 1)

    # Create graphs to display the change in percentage of items not found
    # when changing the hash functions/flipping possibilities.
    plt.title(experiment_name.capitalize() + " Hash Experimentation")
    plt.plot(hash_values, hash_mean, label = "Hash Results")
    plt.plot(hash_values, hash_mean_all_features, label = "Hash Results All Features")
    plt.xlabel("No. of Hash Functions Used")
    plt.ylabel("Percentage of Items not Found")
    plt.legend()
    plt.show()

    plt.title( experiment_name.capitalize() + " Flip Experimentation")
    plt.plot(bit_flip_values, flip_mean, label = "Flip Results")
    plt.plot(bit_flip_values, flip_mean_all_features, label = "Flip Results All Features")
    plt.xlabel("Probability of Bit flip Used")
    plt.ylabel("Percentage of Items not Found")
    plt.legend()
    plt.show()

def run_experiments_bloom_blip(bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs, experiment_name, bloom_and_blip = False):
    # Need to compare the results with normal bloom filters and flipped.

    experiment_dict = {
        'faker': faker_experiment_blip,
        'homogenous': homogenous_experiment_blip,
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
    hash_results_flipped = []
    hash_results = []
    for hash in range(len(hash_values)):
        results_flipped = []
        results = []
        for _ in range(number_of_runs):

            results_flipped.append(test_bloom_filter_content(experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])))
            results.append(test_bloom_filter_content(experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[hash], bit_flip_values[0], bloom_and_blip), bloom_and_blip))

        hash_results_flipped.append(results_flipped)
        hash_results.append(results)

    hash_mean_flipped = np.mean(hash_results_flipped, axis = 1)
    hash_mean = np.mean(hash_results, axis = 1)

    # Create graphs to display the change in percentage of items not found
    # when changing the hash functions/flipping possibilities.
    plt.title(experiment_name.capitalize() + " Hash Experimentation")
    plt.plot(hash_values, hash_mean, label = "Hash Results")
    plt.plot(hash_values, hash_mean_flipped, label = "Hash Results Flipped")
    plt.xlabel("No. of Hash Functions Used")
    plt.ylabel("Percentage of Items not Found")
    plt.legend()
    plt.ylim(bottom = -1, top = 100)
    plt.show()

def run_experiments_profile_reconstruction_attack(bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs, experiment_name):

    experiment_dict = {
        'faker': faker_experiment_blip,
        'homogenous': homogenous_experiment_blip,
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


    def compare_profile_guesses(bloom_data, experiment_name, bit_flip_value):
        results = []

        bloom_dict = bloom_data[0]
        profile_array = bloom_data[1]

        bloom_list = list(bloom_dict.values())
        for bloom in bloom_list:

            profile = profile_array[bloom_list.index(bloom)]
            guesses = list(profile_reconstruction_attack(bloom, experiment_name, bloom.k, bit_flip_value))[0]
            filtered_guesses = list(filter(lambda x: x != False, guesses))
            no_correct_guesses = len([guess for guess in filtered_guesses if guess in profile])
            filtered_profile = list(filter(lambda x: x != None, profile))
            results.append(no_correct_guesses / len(filtered_profile))
            print("Results {}".format(results))
        return results

    # Compare results when different no. of hash functions are used.
    hash_results = []
    for hash in range(len(hash_values)):
        
        for _ in range(number_of_runs):
            bloom_data = experiment_dict[experiment_name](bloom_size, dataset_size, hash_values[hash], bit_flip_values[0])
            # results.append(test_bloom_filter_content(bloom_data))

            hash_results.append(compare_profile_guesses(bloom_data, experiment_name, bit_flip_values[0]))
        print(hash_results)

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

            # results.append(test_bloom_filter_content((bloom_dict, profile_array)))
        bloom_data = (bloom_dict, profile_array)
        flip_results.append(compare_profile_guesses(bloom_data, experiment_name, bit_flip_values[flip]))
        print(hash_results)

    hash_mean = np.mean(hash_results, axis = 1)
    flip_mean = np.mean(flip_results, axis = 1)

    # Create graphs to display the change in percentage of items guessed by the attack
    # when changing the hash functions/flipping possibilities.
    plt.title(experiment_name.capitalize() + " Hash Experimentation")
    plt.plot(hash_values, hash_mean, label = "Hash Results")
    plt.xlabel("No. of Hash Functions Used")
    plt.ylabel("Percentage of Items Guessed")
    plt.show()

    plt.title(experiment_name.capitalize() + " Flip Experimentation")
    plt.plot(bit_flip_values, flip_mean, label = "Flip Results")
    plt.xlabel("Probability of Bit flip Used")
    plt.ylabel("Percentage of Items Guessed")
    plt.show()

# Found the code at the link below and modified it.
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py

def bench_clustering_algorithms(cluster_algorithm, title, data, labels):
    """Benchmark to evaluate clustering algorithms.

    Parameters
    ----------
    cluster_algorithm : Clustering algorithm used
        Chosen clustering algorithm that is being measured.
    title : str
        Cluster name and the data the algorithm was applied to.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), cluster_algorithm).fit(data)
    fit_time = time() - t0

    # Setting up the results and the corresponding output.
    results = [title, fit_time]
    output_string = "\n\tTitle\t\tFit-Time\t"
    format_string = "{:9s}\t{:.3f}s\t\t"

    if title[:title.find(" ")] == "kmeans":
        results += [estimator[-1].inertia_]
        output_string += "Inertia\t"
        format_string += "{:.0f}\t"

    # Define the metrics which require only the true labels and estimator
    # labels
    # clustering_metrics = [
    #     metrics.homogeneity_score,
    #     metrics.completeness_score,
    #     metrics.v_measure_score,
    #     metrics.adjusted_rand_score,
    #     metrics.adjusted_mutual_info_score,
    # ]
    # results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        silhouette_score(
            data,
            estimator[-1].labels_,
            metric = "euclidean",
            sample_size = len(data)
        )
    ]

    output_string += "Silhouette Coefficient\t"
    format_string += "{:.3f}\n"
    print(output_string + "\n")

    # Show the results
    formatter_result = ( format_string )
    print(formatter_result.format(*results))

def run_experiments_clustering(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility, cluster_name, n_clusters = 2, clustering_labels = ""):

    algorithm_dict = {
        'kmeans': KMeans,
        'agglo': AgglomerativeClustering,
        'spectral': SpectralClustering
    }

    # If the inputted experiment isn't in the dictionary then use KMeans 
    # as a default.
    if cluster_name not in algorithm_dict.keys():
        cluster_name = 'kmeans'
    
    # Set the parameters of KMeans or SpectralClustering.
    if cluster_name == "kmeans":
        cluster_algorithm = algorithm_dict[cluster_name](n_clusters = n_clusters, random_state = 0)
    
    elif cluster_name == "spectral":
        cluster_algorithm = algorithm_dict[cluster_name](assign_labels = clustering_labels, n_clusters = n_clusters, random_state = 0)

    else:
        cluster_algorithm = algorithm_dict[cluster_name]()
    
    def perform_zoo_clustering(df, title):

        bench_clustering_algorithms(cluster_algorithm = cluster_algorithm, title = title, data = df, labels = df.columns)

        df['cluster'] = cluster_algorithm.fit_predict(df)
        df["animal_name"] = animals

        animal_cluster_values = [df[df['cluster'] == x ]["animal_name"] for x in range(n_clusters)]

        plt.title(title)
        plt.scatter(df["cluster"], df['animal_name'], c = cluster_algorithm.labels_.astype(float), s=50, alpha=0.5)
        plt.xlabel("cluster")
        plt.ylabel("class_type")
        plt.show()
    
    # Test clustering on the raw data.
    df = pd.read_csv('zooData/zoo.csv')
    df = df[:10]
    animals = df["animal_name"]
    df = df[list(df.columns)[1:]].copy()
    title = cluster_name + " Zoo Raw Data"
    perform_zoo_clustering(df, title)

    # Test clustering on the bloomed data.
    bloom_data = zoo_experiment_blip(bloom_size, dataset_size, number_of_hashes, bit_flip_possibility, True)

    bloom_list = list(bloom_data[0].values())
    bloom_matrix = np.matrix([bloom_list[0].bloomFilter, bloom_list[1].bloomFilter])

    bloom_list_flipped = list(bloom_data[1].values())
    bloom_matrix_flipped = np.matrix([bloom_list_flipped[0].bloomFilter, bloom_list_flipped[1].bloomFilter])

    # Turn the underlying data structure of each Bloom into a numpy array.
    # Create a matrix from the joining of these arrays.
    for bloom in range(len(bloom_list[2:])):

        bloom_list[bloom].list_to_array()
        bloom_list_flipped[bloom].list_to_array()

        bloom_matrix = np.append(bloom_matrix, [bloom_list[bloom].bloomFilter], axis = 0)
        bloom_matrix_flipped = np.append(bloom_matrix_flipped, [bloom_list_flipped[bloom].bloomFilter], axis = 0)

    # Test clustering on the bloomed data.
    df = pd.DataFrame(bloom_matrix)
    title = cluster_name + " Zoo Bloomed Data"
    perform_zoo_clustering(df, title)

    # Test clustering on the blipped data.
    df_flipped = pd.DataFrame(bloom_matrix_flipped)
    title = cluster_name + " Zoo Blipped Data"
    perform_zoo_clustering(df_flipped, title)


if __name__ == "__main__":

    states = """
        Alabama Alaska Arizona Arkansas California Colorado Connecticut
        Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas
        Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota
        Mississippi Missouri Montana Nebraska Nevada NewHampshire NewJersey
        NewMexico NewYork NorthCarolina NorthDakota Ohio Oklahoma Oregon
        Pennsylvania RhodeIsland SouthCarolina SouthDakota Tennessee Texas Utah
        Vermont Virginia Washington WestVirginia Wisconsin Wyoming
        """.split()

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
    # zoo_experiment_blip(30, 10, 2, 0.5)
    
    # # Params: bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs
    # run_experiments(50, 10, list(range(2, 8)), [0.0, 0.5, 0.8, 1.0], 10, "zoo")
    # run_experiments(100, 50, list(range(21, 26)), [0.25, 0.5, 0.75, 0.9, 1.0], 10, "zoo")
    # Flip value is 0.0

    # # Params: bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs
    # run_experiments_all_features(50, 10, list(range(2, 8)), [0.2, 0.5, 0.8, 1.0], 10, "zoo")
    # run_experiments_all_features(100, 50, list(range(21, 26)), [0.25, 0.5, 0.75, 0.9, 1.0], 10, "zoo")

    # # Params: bloom_size, dataset_size, hash_values, bit_flip_values, number_of_runs, bloom_and_blip
    # run_experiments_bloom_blip(50, 10, list(range(2, 8)), [0.2, 0.5, 0.8, 1.0], 10, "zoo", True)
    # run_experiments_bloom_blip(100, 50, list(range(21, 26)), [0.25, 0.5, 0.75, 0.9, 1.0], 10, "zoo", True)

    # # Params: bloom_size, dataset_size, number_of_hashes, bit_flip_possibility = 0.5, bloom_and_blip = False
    # run_experiments(50, 10, list(range(2, 8)), [0.2, 0.5, 0.8, 1.0], 10, "homogenous")
    # run_experiments(100, 50, list(range(21, 26)), [0.25, 0.5, 0.75, 0.9, 1.0], 10, "homogenous")

    # # Creating Bloom filters to test the Profile Reconstruction Attack on

    # bloom1 = BloomFilter(10, 2)
    # bloom1.add("hair1")
    # print_bloom_filter(bloom1)
    # bloom1.add("class_type1")
    # print_bloom_filter(bloom1)
    # bloom1 = bit_flipping(bloom1, 0.2)
    # print_bloom_filter(bloom1)

    # print(profile_reconstruction_attack(bloom1, "zoo", 0.2))

    # bloom1 = BloomFilter(10, 2)
    # bloom1.add("tkhab")
    # print_bloom_filter(bloom1)
    # bloom1.add("pqofj")
    # print_bloom_filter(bloom1)
    # bloom1 = bit_flipping(bloom1, 0.2)
    # print_bloom_filter(bloom1)

    # print(profile_reconstruction_attack(bloom1, "homogenous", bloom1.k, 0.0, 5))

    # Need to compare the items guessed by the profile reconstruction attack and the items in 
    # the original bloom to get the percentage of items that were correctly guessed. 

    # run_experiments_profile_reconstruction_attack(50, 10, list(range(2, 8)), [0.2, 0.5, 0.8, 1.0], 3, "zoo")
    # run_experiments_profile_reconstruction_attack(100, 50, list(range(21, 26)), [0.25, 0.5, 0.75, 0.9, 1.0], 10, "zoo")

    run_experiments_clustering(50, 10, 2, 0.2, "kmeans", 3)
    # run_experiments_clustering(50, 10, 2, 0.2, "kmeans", 7)
    run_experiments_clustering(50, 10, 2, 0.2, "agglo")
    # run_experiments_clustering(50, 10, 2, 0.2, "spectral", 3, "kmeans")
    # run_experiments_clustering(50, 10, 2, 0.2, "spectral", 3, "discretize")
    run_experiments_clustering(50, 10, 2, 0.2, "spectral", 3, "cluster_qr")


    # Sort the profile reconstruction attack experiments -> look at the hash_results and flip_results
    # created and see if they're the correct shape. See if the graph will plot.

    # Need to fix the memory error issue with the homogenous dataset. Look at the recommendation from mentor.

    # Need to try to get the bloom filters to work with the clustering algorithms of scikit-learn.

    