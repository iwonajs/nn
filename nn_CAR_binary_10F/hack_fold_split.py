import json
import random

folds10_pos = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}}
folds10_neg = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}}
#fold_flag = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

fold_flag = {0: False, 1: False, 2: False, 3: False, 4: False}
bugFold = {}
idlist = []
#print(fold_flag)
fold10_pos = [0 for i in range(10)]
fold10_neg = [0 for i in range(10)]
fold5_pos = [0 for i in range(5)]
fold5_neg = [0 for i in range(5)]

with open('folds_pos_neg.json', 'r') as folds5json_file:
    folds5 = json.load(folds5json_file)

for key, data in folds5.items():
    #print(key)
    print(data)
    #print(fold_flag.get(0))
    fold_id = data['fold']
    bug_id = data['bug']
    idlist.append(data['id'])
    if bug_id in bugFold:
        fold_id_new = bugFold[bug_id]
    else:
        # flip or not flip?
        if fold_flag[fold_id]:
            fold_id_new = fold_id + 5
            fold_flag[fold_id] = False
        else:
            fold_id_new = fold_id
            fold_flag[fold_id] = True
        bugFold[bug_id] = fold_id_new

    data['fold'] = fold_id_new
    obs_id = data['id']
    if data['pos']:
        folds10_pos[fold_id_new][obs_id] = data
        if not (fold_id_new == folds10_pos[fold_id_new][obs_id]['fold']):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ssssssss")
    else:
        folds10_neg[fold_id_new][obs_id] = data


    if data['pos']:
        fold5_pos[fold_id] += 1
        fold10_pos[fold_id_new] += 1
    else:
        fold5_neg[fold_id] += 1
        fold10_neg[fold_id_new] +=1


    #folds10[key] = data
    #if data['bug']=='1909':
    #    print(data)
    #exit(1)
print(fold5_pos)
print(fold5_neg)
print(fold10_pos)
print(fold10_neg)
idlist = sorted(set(idlist))
print(len(idlist), idlist)
print(sum(fold5_pos)*2)

for i in range(5):
    bucket_pos = fold10_pos[i]
    bucket_neg = fold10_neg[i]
    bucket_pos_s = fold10_pos[i+5]
    bucket_neg_s = fold10_neg[i+5]
    bucket_neg_diff = bucket_neg - bucket_pos
    print("------------", i, i+5)
    print(bucket_pos, bucket_neg, bucket_pos - bucket_neg)
    print(bucket_pos_s, bucket_neg_s, bucket_pos_s - bucket_neg_s)
    print(bucket_neg_diff)

    # NEG BUCKET IS TOO SMALL for bucket i
    if bucket_neg_diff < 0:
        counter = 0
        pop = []
        possible_bug_ids = [bug for bug, fold in bugFold.items() if fold == i]
        print("xxx", len(possible_bug_ids), possible_bug_ids)
        for key, data in folds10_neg[i+5].items():
            if counter < abs(bucket_neg_diff):
                counter += 1
                data['bug'] = random.choice(possible_bug_ids)
                data['fold'] = i
                obs_id = data['id']
                folds10_neg[i][obs_id] = data
                pop += [key]
        for pi in range(len(pop)):
            folds10_neg[i+5].pop(pop[pi])

        if counter == abs(bucket_neg_diff):
            print("Done!", i)
        else:
            print("PROBLEM!!!!!!!!!!!!", i)

    # NEG BUCKET IS TOO BIG for bucket i
    if bucket_neg_diff > 0:
        counter = 0
        pop = []
        possible_bug_ids = [bug for bug, fold in bugFold.items() if fold == i]
        print("qqqq", len(possible_bug_ids), possible_bug_ids)
        for key, data in folds10_neg[i].items():
            if counter < abs(bucket_neg_diff):
                counter += 1
                data['bug'] = random.choice(possible_bug_ids)
                data['fold'] = i+5
                obs_id = data['id']
                folds10_neg[i+5][obs_id] = data
                pop += [key]
        for pi in range(len(pop)):
            folds10_neg[i].pop(pop[pi])
        if counter == abs(bucket_neg_diff):
            print("Done!", i)
        else:
            print("PROBLEM!!!!!!!!!!!!", i)


for i in range(10):
    p = len(folds10_pos[i].items())
    n = len(folds10_neg[i].items())
    print("xxxxxxxxxxxx",p , n, p+n)

folds10 = {}
for fold, data in folds10_pos.items():
    print(fold, len(data)*2, data)
    for id, obs in data.items():
        if not (obs['fold'] == fold):
            exit(1)
        folds10[id] = obs
for fold, data in folds10_neg.items():
    print(fold, len(data)*2, data)
    for id, obs in data.items():
        if not (obs['fold'] == fold):
            exit(2)
        folds10[id] = obs

#folds10.update(folds10_neg)

with open('folds_10_pos_neg.json', 'w') as outfile:
    json.dump(folds10, outfile)

check = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
with open('folds_10_pos_neg.json', 'r') as testfile:
    folds = json.load(testfile)

for k, v in folds.items():
    fold_id = v['fold']
    check[fold_id] += 1
print(check)