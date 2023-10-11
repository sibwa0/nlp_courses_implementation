words_lst = [
    ('в', 10),
    ('не', 9),
    ('это', 9),
    ('а', 8),
    ('кпрф', 7),
    ('на', 6),
    ('и', 5),
    ('свой', 5),
    ('все', 4),
    ('быть', 4),
    ('что', 4),
    ('они', 3),
    ('вот', 3),
    ('он', 3),
    ('пост', 3),
    ('с', 3),
    ('то', 3),
    ('чтобы', 3),
    ('как', 2),
    ('именно', 2),
    ('по', 2),
    ('но', 2),
    ('еще', 2),
    ('бизнес', 2),
    ('прогорать', 2)]

start = 0
end = len(words_lst) - 1
index = -1
min_count = 7
cur = -1


if __name__ == "__main__":
    while cur != min_count:
        index = start + (end - start) // 2
        cur = words_lst[index][1]
        if cur < min_count:
            end = index
        elif cur > min_count:
            start = index + 1
    

    next_word_count = words_lst[index + 1][1]
    while next_word_count == min_count:
        index += 1
        if index == end:
            break
        next_word_count = words_lst[index + 1][1]

    print(words_lst[index], index)
