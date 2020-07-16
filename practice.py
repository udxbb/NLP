# def intersect(a, b):
#     if len(a) == 1 and len(b) == 1:
#         if a[0] == b[0]:
#             result.append(a[0])
#         return 1
#     else:
#         if len(a) != 1:
#             a1 = a[:len(a) // 2]
#             a2 = a[len(a) // 2:]
#             intersect(a1, b)
#             intersect(a2, b)
#         else:
#             b1 = b[:len(b) // 2]
#             b2 = b[len(b) // 2:]
#             intersect(a, b1)
#             intersect(a, b2)
#         return result

#
# result = []
#
#
# def intersect(a, b):
#     if len(a) == 1 and len(b) == 1:
#         if a == b:
#             result.append(a[0])
#         return 1
#     else:
#         if len(a) != 1:
#             a1 = a[:len(a) // 3]
#             a2 = a[len(a) // 3: (len(a)//3) * 2]
#             a3 = a[(len(a)//3) * 2:]
#             intersect(a1, b)
#             intersect(a2, b)
#             intersect(a3, b)
#         else:
#             b1 = b[:len(b) // 3]
#             b2 = b[len(b) // 3: (len(b) // 3) * 2]
#             b3 = b[(len(b) // 3) * 2:]
#             intersect(a, b1)
#             intersect(a, b2)
#             intersect(a, b3)
#         return result

test = [1, 2, 3, 4]
if 3 < len(test) or 2 < len(test):
    print(1)


def union(a, b):
    result_union = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result_union.append(a[i])
            i += 1
            j += 1
        elif a[i] > b[j]:
            result_union.append(b[j])
            j += 1
        else:
            result_union.append(a[i])
            i += 1
    return result_union


def a_and_not_b(a, b):
    result_union = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] > b[j]:
            result_union.append(a[i])
            i += 1
            j += 1
        else:
            result_union.append(a[i])
            i += 1
    return result_union


lis1 = [45, 9, 13, 12]
lis2 = [45, 13, 14, 12]
# result1 = intersect(lis1, lis2)
result_3 = a_and_not_b(lis1, lis2)
# print(result1)
print(result_3)
