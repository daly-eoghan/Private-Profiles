from bloomFilter import BloomFilter
import random

def print_bloom_filter(bloom):

    print("[" + ", ".join([str(bit) for bit in bloom.bloomFilter]) + "]")



def bit_flipping(bloom, p = 0.5):

    for x in range(len(bloom.bloomFilter)):
        # random -> float value between 0 and 1.
        if random.random() > p:
            # Apply an XOR function to the bit value 
            bloom.bloomFilter[x] = bloom.bloomFilter[x] ^ 1

    return bloom

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
    
    # print(len(states))

    # Testing Bloom filter functionality.

    # Takes as arguments the size of the bloom filter and the 
    # number of hash functions used. 
    b1 = BloomFilter(len(states), 2)

    for x in states:
        b1.add(x)
    
    # print(b1.contains("Washington"))
    # print(b1)
    # b1.generateStats()
    # print_bloom_filter(b1)

    # Testing Bit flipping functionality.

    # Try flipping all bits.
    # print("\nTest 1\n")
    # print_bloom_filter(b1)
    # print("\n")
    # bit_flipping(b1, 0)
    # print_bloom_filter(b1)

    # # Try flipping zero bits
    # print("\nTest 2\n")
    # print_bloom_filter(b1)
    # print("\n")
    # bit_flipping(b1, 1)
    # print_bloom_filter(b1)

    