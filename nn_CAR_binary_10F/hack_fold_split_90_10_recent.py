import os
import random
import json
import datetime
from datetime import date
import random

five_fold_json = "/home/iwona/optane/msr_input/folds_pos_neg.json"
# BUG LEDGER: created_at, closed_at
bug_ledger_json = "/home/iwona/optane/Data/bugs/issues_etl_bug_compact.json"

with open("config.json") as f:
    configs = json.load(f)


train_json = configs["train_pos_folds"]
test_json = configs["test_pos"]
print(train_json)
print(test_json)

with open(five_fold_json, 'r') as folds5json_file:
    folds5 = json.load(folds5json_file)

bug_age = {}
with open(bug_ledger_json, 'r') as f:
    for line in f:
        body = json.loads(line)
        date_string = body["created_at"]
        date_string = str(date_string).replace("Z","")
        date_x = datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
        delta = date.today() - date(date_x.year, date_x.month, date_x.day)
        bugid = str(body["gh"]).replace("gh-", "")
        bug_age[bugid] = delta.days
print(bug_age)





    #fold_id = data['fold']
    #idlist.append(data['id'])
    #if bug_id in bugFold:

with open(five_fold_json, 'r') as folds5json_file:
    folds5 = json.load(folds5json_file)

set_bugs = set()
filter_ids = []
#tenPercent = {}
#ninetyPercent = {}
for key, data in folds5.items():
    bug_id = data['bug']
    data["fold"] = -1
    set_bugs.add(bug_id)
    if bug_id in bug_age:
        filter_ids += [bug_id]
    elif not bug_id in bug_age:
        print("missing:", bug_id)
        exit(99)

filter_ids = list(set(filter_ids))
print(len(folds5)/2)
print(len(filter_ids))
print(len(set_bugs))

bug_age_filtered = {}
for x in range(len(filter_ids)):
    bug_id = filter_ids[x]
    bug_age_filtered[bug_id] = bug_age[bug_id]

print(bug_age_filtered)
print("Filtered Bug Age", len(bug_age_filtered))

z = sorted(bug_age_filtered.items(), key=lambda x: x[1])

tenPercent_N = round(len(z)*.1)
tenPercent_bug_age = z[:tenPercent_N]
tenPercent_ids = [item[0] for item in tenPercent_bug_age]

ninteyPercent_bug_age = z[tenPercent_N:]
ninteyPercent_ids = [item[0] for item in ninteyPercent_bug_age]
print("TEN PERCENT ('bug_id', age")
print(tenPercent_ids)

tenPercent = {}
ninetyPercent = {}
negative_obs = {}
count_10 = 0
count_90 = 0
count_neg = 0
for key, data in folds5.items():
    bug_id = data['bug']
    data['fold'] = -1
    if data['pos']:
        if bug_id in tenPercent_ids:
            tenPercent[count_10] = data
            count_10 += 1
        else:
            ninetyPercent[count_90] = data
            count_90 += 1
    else:
        negative_obs[count_neg] = data
        count_neg += 1

print("90% observations count:", len(ninetyPercent))
print("90% bug count:", len(ninteyPercent_ids))
fold_N = round(len(ninteyPercent_ids) / 10)
print("fold bug size:", fold_N)
lastFold = len(ninteyPercent_ids)-9*fold_N
print(lastFold)
fold_bug_size = [fold_N for i in range(9)]
fold_bug_size += [lastFold]
print(fold_bug_size)


folds_ids_lists = []
start = 0
for fold in range(len(fold_bug_size)):
    end = sum(fold_bug_size[0:fold+1])
    fold_ids = ninteyPercent_ids[start:end]
    folds_ids_lists += [fold_ids]
    print(start, end)
    for key, data in ninetyPercent.items():
        bug_id = data['bug']
        if bug_id in fold_ids:
            data['fold'] = fold
            ninetyPercent[key] = data
    start = end

with open(configs['train_pos_folds'], 'w') as f:
    json.dump(ninetyPercent, f)

with open(configs['test_pos'], 'w') as f:
    json.dump(tenPercent, f)

check = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
with open(configs['train_pos_folds'], 'r') as testfile:
    folds = json.load(testfile)

for k, v in folds.items():
    fold_id = v['fold']
    check[fold_id] += [v['bug']]
    print(k, v)
#for k, v in check.items():
#    print("Fold:", k, len(set(v)), len(v))







