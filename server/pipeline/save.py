import re

def count_in_match(vocab, count_dict, trend_list):
    try:
        if vocab in count_dict:
            start_idx, end_idx = 0, len(trend_list)

            if count_dict[vocab] < trend_list.count(vocab):
                rep = count_dict[vocab]
            else:
                rep = trend_list.count(vocab)

            for each in range(rep):
                word_idx = trend_list.index(vocab, start_idx, end_idx)
                start_idx = word_idx + 1

            vocab_idx = word_idx
            count_dict[vocab] += 1
        else: 
            count_dict[vocab] = 1
            vocab_idx = trend_list.index(vocab)
    except ValueError:
        print("Value error!")
        vocab_idx = trend_list.index(vocab)

    return vocab_idx, count_dict

def check_year(data):
    if int(data) <= 2023:
        return True
    return False

def manual_date_detection(date_phrase):
    month = {'jan':'01', 'january':'01', 'feb':'02', 'february':'02', 'mar':'03', 'march':'03', 'apr':'04', 'april':'04', 'may':'05', 'june':'06', 'july':'07', \
             'aug':'08', 'august':'08', 'sep':'09', 'september':'09', 'oct':'10', 'october':'10', 'nov':'11', 'november':'11', 'dec':'12', 'december':'12'}
    
    y, m, d = "XXXX", "XX", "XX"
    for k, v in month.items():
        if re.findall(k, date_phrase):
            m = v
            break
    
    if re.findall("\d{1,2}[a-z]{2}?", date_phrase):
        d = re.findall("\d{1,2}[a-z]{2}?", date_phrase)[0]

    if re.findall("\d{4}", date_phrase):
        y =  re.findall("\d{4}", date_phrase)[0]

    return f"{y}-{m}-{d}"

if __name__ == '__main__':
    d = "february"
    print(manual_date_detection(d))
    d = " march of 2020"
    print(manual_date_detection(d))