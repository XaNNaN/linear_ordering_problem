s = "11954 0 19 604 0 6 0 23 0 716 942 11 25 28 72 28 148 651 0 0 1 0 664 198 1 46 0 100 4 69 0 68 40 0 0 845 0 0 0 0 103 601 66 1418 185 25 1084 135 33 0 162 0 23 451 0 15 573 0 0 0"
numbers = [word for word in s.split() if word.isdigit()]
count = len(numbers)
print(count)  # 3