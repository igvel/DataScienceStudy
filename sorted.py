# Sorted test

data = [['a', 3], ['z', 5], ['d', 1]]

sorted_data = sorted(data, reverse=True)

for item in sorted_data:
    print(item[0], item[1])

