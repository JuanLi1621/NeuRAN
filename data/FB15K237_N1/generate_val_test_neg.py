def run(dir):
    neg_triples = []
    with open(dir) as fin:
        for line in fin.readlines():
            h, t, r, l = line.strip().split()
            if int(l) == -1:
                neg_triples.append((h,t,r))
            elif int(l) == 1:
                continue
            else:
                print('error!')
    return neg_triples

def write_neg(triples, out_path):
    with open(out_path, 'w') as fin:
        fin.write(str(len(triples))+'\n')
        for i in triples:
            fin.write(i[0]+'\t'+i[1]+'\t'+i[2]+'\n')

neg_valid = run('./valid_neg.txt')
write_neg(neg_valid, './valid2id_neg.txt')
neg_test = run('./test_neg.txt')
write_neg(neg_test, './test2id_neg.txt')